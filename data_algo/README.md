In the three algo sub-folders, please load your bus route output for the evaluation process.

Save it with the following naming convention, either as `.csv` or `.xlsx`:

- `schedule_SA`
- `schedule_LP`
- `schedule_GA`

Your excel must contain the following columns:

- `trip_id`: This is the unique trip ID
- `route_short_name`: This is the route number
- `trip_headsign`: E.g. Destination
- `stop_sequence`: starting with "0", counting the order in which the bus stop is driven to along the trip.
- `stop_name`: Name of bus stop
- `minutes_to_stop`: Estimated minutes to bus stop, from previous one
- `arrival_time`: Estimated arrival time on bus stop
- `stop_lat`: Latitude coordinate of bus stop
- `stop_lon`: Longitude coordinate of bus stop

See *base_n_evaluation/output/hvv_total_schedule.xlsx* for an example.