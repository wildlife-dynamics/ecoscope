import ecoscope
import time
import asyncio
import pandas as pd
import os

ER_SERVER = "https://mep-dev.pamdas.org"
ER_USERNAME = os.getenv("ER_USERNAME")
ER_PASSWORD = os.getenv("ER_PASSWORD")

since = pd.Timestamp("2000-07-01").isoformat()
# since = pd.Timestamp("2024-06-25").isoformat()
# since = pd.Timestamp("2024-07-01").isoformat()

client = ecoscope.io.EarthRangerIO(server=ER_SERVER, username=ER_USERNAME, password=ER_PASSWORD)
as_client = ecoscope.io.AsyncEarthRangerIO(server=ER_SERVER, username=ER_USERNAME, password=ER_PASSWORD)

start = time.time()
filtered_df = client.get_patrols(since=since)
obs = client.get_patrol_observations(filtered_df)
print(f"synchronous get subjects took {time.time()-start}s")

start = time.time()


async def test():
    result = await as_client.get_patrol_observations(since=since)
    await as_client.close()
    return result


observations = asyncio.run(test())
# observations = asyncio.gather(test())
print(f"asynchronous get subjects took {time.time()-start}s")

print(len(obs))
print(len(observations))
