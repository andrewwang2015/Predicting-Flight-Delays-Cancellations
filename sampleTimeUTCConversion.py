import datetime
import pytz
from tzwhere import tzwhere

tzwhere = tzwhere.tzwhere()
timezone_str = tzwhere.tzNameAt(34.1478, -118.1445) # Pasadena

print(timezone_str)


timezone = pytz.timezone(timezone_str)
dt = datetime.datetime.now()
print(dt)
current = timezone.localize(dt)
currentUTC = current.astimezone(pytz.utc)
print(currentUTC)

