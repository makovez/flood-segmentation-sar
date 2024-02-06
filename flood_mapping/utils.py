from datetime import datetime
def today():
    # Get the current timestamp as a datetime object
    current_time = datetime.now()

    # Convert the datetime object to a string
    current_time_str = current_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    return current_time_str