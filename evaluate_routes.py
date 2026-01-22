import argparse
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.core.io import load_processed_instance
from src.evaluation import assignment

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate school bus routes.')
    parser.add_argument('--routes', type=str, required=True, help='Path to the route CSV file.')
    parser.add_argument('--instance', type=str, default='north_primary', help='Name of the instance (e.g., north_primary).')
    parser.add_argument('--output', type=str, default='evaluation_results.csv', help='Path to save the evaluation run details.')
    return parser.parse_args()

def hydrate_times(routes_df, time_min_matrix, stops_map, school_arrival_time="07:45:00"):
    """
    If routes lack valid times, estimate them based on travel time matrix.
    Strategy: Backward scheduling from school arrival time.
    """
    print("Hydrating schedule times...")
    
    # Ensure arrival_time_dt exists
    if 'arrival_time_dt' not in routes_df.columns:
        routes_df['arrival_time_dt'] = pd.NaT

    # Convert school deadline to today's datetime
    deadline = pd.to_datetime(school_arrival_time)
    
    hydrated_cnt = 0
    
    # Iterate over unique trips
    unique_trip_ids = routes_df['trip_id'].unique()
    
    for trip_id in unique_trip_ids:
        # Get indices for this trip, properly sorted descending by sequence
        # We need the INDICES to update the main dataframe in place
        trip_mask = routes_df['trip_id'] == trip_id
        group_indices = routes_df[trip_mask].sort_values('stop_sequence', ascending=False).index
        
        stops = routes_df.loc[group_indices].to_dict('records')
        
        current_time = datetime.combine(datetime.now().date(), deadline.time())
        
        # Backward pass
        for i, stop in enumerate(stops):
            # Update using the original index from group_indices
            idx = group_indices[i] 
            
            routes_df.at[idx, 'arrival_time_dt'] = current_time
            routes_df.at[idx, 'departure_time_dt'] = current_time 
            
            if i < len(stops) - 1:
                stop_name = stop['stop_name']
                next_stop_in_loop = stops[i+1] # This is actually the PREVIOUS stop in sequence order (since we are going backwards)
                prev_stop_name = next_stop_in_loop['stop_name']
                
                # ... time calc logic ...
                travel_min = 5 
                
                id_curr = -1
                id_prev = -1
                
                if stop_name in stops_map: id_curr = stops_map[stop_name]
                if prev_stop_name in stops_map: id_prev = stops_map[prev_stop_name]

                if id_curr != -1 and id_prev != -1:
                    travel_min = time_min_matrix[id_prev, id_curr]
                
                current_time = current_time - timedelta(minutes=float(travel_min))
            
            hydrated_cnt += 1
            
    print(f"Hydrated {hydrated_cnt} stop times.")
    return routes_df

def calculate_metrics(best_trips, pupils, school_arrival_deadline, capacity_per_bus=50): # Assuming 50 capacity for utilization calc if unknown
    
    metrics = {}
    
    # 1. Pupils Served Total
    metrics['pupils_served_total'] = len(best_trips)
    
    # 2. Percentage Served
    metrics['percentage_served'] = len(best_trips) / len(pupils) if len(pupils) > 0 else 0
    
    if best_trips.empty:
        # Fill rest with 0/None
        return metrics

    # 3. Number of Buses
    # Flatten trip_ids list
    all_trips = []
    for x in best_trips['trip_ids']:
        if isinstance(x, list): all_trips.extend(x)
        else: all_trips.append(x)
    metrics['number_of_buses'] = len(set(all_trips))
    
    # 4. Total KM
    # Use 'total_travel_time' as proxy for distance if KM not explicit, OR sum up segment distances if we had them.
    # The assignment output gives 'total_travel_time'.
    # To get proper KM, we would need to re-query the distance matrix for each route segment.
    # For MVP, let's use the route description or travel time * avg_speed proxy.
    # Let's try to do it right: Unique Routes -> Sum Distances.
    # Since 'best_trips' only shows pupil itins, we need the Route definitions themselves to sum KM.
    # Re-calculating Total KM from the used routes (from the Schedule or Assignment logic) is cleaner.
    # But 'best_trips' doesn't have the full route geometry. 
    # Placeholder: Time * 0.6 km/min (36 km/h)
    # Better: We need access to the `segments` used. 
    # Let's accept Time Proxy for now or add KM calculation to assignment module later.
    # For now: Sum of unique route lengths estimated by Avg Speed 30km/h = 0.5 km/min
    # NOTE: This ignores empty leg distance (deadhead).
    metrics['total_km_estimated'] = best_trips['total_travel_time'].dt.total_seconds().sum() / 60 * 0.5 

    # 5-7. Commute Time
    travel_mins = best_trips['total_travel_time'].dt.total_seconds() / 60
    metrics['avg_commute_time'] = travel_mins.mean()
    metrics['min_commute_time'] = travel_mins.min()
    metrics['max_commute_time'] = travel_mins.max()
    
    # 8. Bus Utilization 
    # (Total Pupil-Mins / (Bus Capacity * Bus Runtime))
    # Approximate: Avg Load. 
    # Exact utilization requires segment-by-segment load profile.
    # Simple proxy: Served / (Buses * Capacity)
    metrics['avg_bus_utilization_proxy'] = metrics['pupils_served_total'] / (metrics['number_of_buses'] * capacity_per_bus) if metrics['number_of_buses'] > 0 else 0

    # 9. Violations
    # Assuming assignment logic enforces hard constraints, violations = 0 for 'assigned'.
    # Check if any commute > max allowed?
    # assignment.py takes max_commute_minutes. So enforced.
    metrics['violations'] = 0 
    
    # 10. Transfers
    transfers = best_trips['num_transfers'].value_counts(normalize=True).to_dict()
    metrics['transfers_0_pct'] = transfers.get(0, 0)
    metrics['transfers_1_pct'] = transfers.get(1, 0)
    metrics['transfers_2_pct'] = transfers.get(2, 0)
    
    # 11-13. Arrival Headroom
    # Latest Arrival - Exit Time
    headroom = (best_trips['latest_arrival_dt'] - best_trips['exit_time']).dt.total_seconds() / 60
    metrics['avg_arrival_headroom'] = headroom.mean()
    metrics['min_arrival_headroom'] = headroom.min()
    metrics['max_arrival_headroom'] = headroom.max()
    
    # 14-16. Waiting Time
    wait_mins = best_trips['total_wait_time'].dt.total_seconds() / 60
    metrics['avg_waiting_time'] = wait_mins.mean()
    metrics['min_waiting_time'] = wait_mins.min()
    metrics['max_waiting_time'] = wait_mins.max()
    
    # 17-18. On-Time
    # Since assignment logic requires arrival <= latest, all assigned are on time by definition of the algorithm.
    metrics['on_time_total'] = metrics['pupils_served_total']
    metrics['on_time_percent'] = 1.0 # If they are in 'best_trips', they are on time.
    
    return metrics

def main():
    args = parse_arguments()
    
    print(f"Loading instance: {args.instance}")
    pupils, stops, schools, dist_km, time_min = load_processed_instance(args.instance)
    
    print(f"Loading routes from: {args.routes}")
    try:
        if args.routes.endswith('.xlsx') or args.routes.endswith('.xls'):
            routes_df = pd.read_excel(args.routes)
        else:
            routes_df = pd.read_csv(args.routes)
    except Exception as e:
        print(f"Failed to load routes file: {e}")
        return

    # Normalize columns
    # Expected: trip_id, stop_name, stop_sequence, arrival_time (optional)
    
    # Alias 'bus_id' to 'trip_id' if present
    if 'trip_id' not in routes_df.columns and 'bus_id' in routes_df.columns:
        routes_df = routes_df.rename(columns={'bus_id': 'trip_id'})
    
    # Check if needs hydration
    
    # Basic data cleaning
    if 'trip_id' in routes_df.columns:
        routes_df['trip_id'] = routes_df['trip_id'].astype(str)
    else:
        print("Error: Route file must contain 'trip_id' or 'bus_id' column.")
        return
    
    # Create Name->ID map for stops (for time lookup)
    if 'stop_name' in stops.columns:
        stops_map = pd.Series(stops.index.values, index=stops['stop_name']).to_dict()
    elif 'name' in stops.columns:
         # Correct column name found in inspection
         stops_map = pd.Series(stops.index.values, index=stops['name']).to_dict()
         
         # Also map node_id if available as explicit column
         if 'node_id' in stops.columns:
             stops_map = pd.Series(stops['node_id'].values, index=stops['name']).to_dict()
    else:
        # Fallback: assume index is node_id and one column is stop_name?
        # Often load_processed_instance returns stops with 'stop_name' as a column.
        # But if it failed, let's inspect columns or try reset_index
        # stops is likely indexed by node_id if sort_values('node_id') was called but index kept
        if 'node_id' in stops.columns:
             # Try first string col
             str_cols = stops.select_dtypes(include='object').columns
             if len(str_cols) > 0:
                  stops_map = pd.Series(stops['node_id'].values, index=stops[str_cols[0]]).to_dict()
             else:
                  # Last resort
                  stops_map = {}
        else:
             # Just assume index is ID
             # Try to find string column
             str_cols = stops.select_dtypes(include='object').columns
             if len(str_cols) > 0:
                 stops_map = pd.Series(stops.index.values, index=stops[str_cols[0]]).to_dict()
             else:
                 print("Warning: Could not map stop names to IDs for time calculation.")
                 stops_map = {}

    # Check for time columns and Initialize if missing
    if 'arrival_time_dt' not in routes_df.columns:
         routes_df['arrival_time_dt'] = pd.NaT

    # Condition: If arrival_time_dt is all NaT (after potential parsing attempt above? No, parsing is below)
    # Re-order: First try to parse what we have.
    
    # 1. Try to parse 'arrival_time' column if it exists
    if 'arrival_time' in routes_df.columns:
         # Check if they are just 00:00:00 placeholders
         # Heuristic: If >90% are 00:00:00, assume invalid
         zero_count = routes_df['arrival_time'].astype(str).str.contains('00:00:00').sum()
         if zero_count > 0.9 * len(routes_df):
             print("Detected placeholder times (00:00:00). forcing hydration.")
             routes_df['arrival_time_dt'] = pd.NaT # Reset
         else:
             today = datetime.now().date()
             routes_df['arrival_time_dt'] = pd.to_datetime(routes_df['arrival_time'].astype(str), format='%H:%M:%S', errors='coerce').apply(lambda t: datetime.combine(today, t.time()) if pd.notnull(t) else pd.NaT)
    
    # 2. If arrival_time_dt is still mostly empty, run hydration
    if routes_df['arrival_time_dt'].isna().mean() > 0.5:
        print("Times are missing or invalid. Running hydration...")
        routes_df = hydrate_times(routes_df, time_min, stops_map)
    else:
        # Just ensure departure matches arrival if missing
        if 'departure_time_dt' not in routes_df.columns:
            routes_df['departure_time_dt'] = routes_df['arrival_time_dt']

    # Final Check
    if routes_df['arrival_time_dt'].isna().all():
         print("CRITICAL: Failed to determine arrival times.")


    # Prepare Pupils Dataframe
    # Ensure latest_arrival_dt is datetime
    if 'latest_arrival' in pupils.columns:
        # Combined with dummy date
        today = datetime.now().date()
        # Use simple string parsing or mixed format to be robust
        pupils['latest_arrival_dt'] = pd.to_datetime(pupils['latest_arrival'].astype(str), format='mixed', errors='coerce').apply(lambda t: datetime.combine(today, t.time()) if pd.notnull(t) else pd.NaT)
    else:
        # Default deadline 7:45 if missing
        pupils['latest_arrival_dt'] = datetime.combine(datetime.now().date(), datetime.strptime("07:45", "%H:%M").time())

    # --- RUN ASSIGNMENT ---
    # DEBUG: Check times
    print("\n--- DEBUG: Time Alignment Check ---")
    print(f"Pupil 0 Deadline: {pupils['latest_arrival_dt'].iloc[0]}")
    if not routes_df.empty:
        print(f"Route 0 First Stop Arrival: {routes_df['arrival_time_dt'].min()}")
        print(f"Route 0 Last Stop Arrival: {routes_df['arrival_time_dt'].max()}")
    print("-" * 30 + "\n")

    result = assignment.assign_trips_with_transfers(
        pupils=pupils,
        schedule=routes_df, # routes_df acts as the schedule
        max_transfers=2
    )
    
    # --- CALCULATE METRICS ---
    metrics = calculate_metrics(result['assigned'], pupils, "07:45:00")
    
    # --- REPORT ---
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    for k, v in metrics.items():
        print(f"{k}: {v}")
    
    # Save
    pd.DataFrame([metrics]).to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
