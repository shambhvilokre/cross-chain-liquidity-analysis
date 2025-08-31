import requests
import pandas as pd

url = "http://localhost:8000/graphql"
payload = { #TODO Query based on schema
    "query": """
query MyQuery {
  burns(
    first: 1000
    skip: 0
    orderBy: timestamp
    orderDirection: asc
    where: {timestamp_gte: "1751328000", timestamp_lt: "1754006400", pool: "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"}
  ) {
    id
    timestamp
    owner
    amount
    amount0
    amount1
    tickLower
    tickUpper
    origin
    transaction {
      id
      gasPrice
      gasUsed
      blockNumber
    }
    logIndex
  }
  swaps(
    first: 1000
    skip: 0
    orderBy: timestamp
    orderDirection: asc
    where: {timestamp_gte: "1751328000", timestamp_lt: "1754006400", pool: "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"}
  ) {
    id
    origin
    recipient
    sender
    timestamp
    amount0
    amount1
    amountUSD
    tick
    sqrtPriceX96
    transaction {
      id
      gasPrice
      gasUsed
      blockNumber
    }
    logIndex
  }
  mints(
    first: 1000
    skip: 0
    orderBy: timestamp
    orderDirection: asc
    where: {timestamp_gte: "1751328000", timestamp_lt: "1754006400", pool: "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"}
  ) {
    amount
    amount0
    amount1
    id
    owner
    sender
    origin
    tickLower
    tickUpper
    transaction {
      id
      gasUsed
      gasPrice
      blockNumber
    }
    timestamp
    amountUSD
    logIndex
  }
}
"""
}
response = requests.post(url, json=payload)
response_json = response.json()
if response_json is None:
  print("Error: response_json is None. Raw response content:")
  print(response.text)
else:
  print("Response keys:", response_json.keys())
print("Data keys:", response_json.get("data", {}).keys())
# Debug: print length and sample of each key's data
if "data" in response_json:
  for key, value in response_json["data"].items():
    if isinstance(value, list):
      print(f"Key: {key}, Length: {len(value)}, Sample: {value[:1]}")
    else:
      print(f"Key: {key}, Type: {type(value)}, Value: {value}")

# Combine all lists from each key in response_json['data']

# Add a column to indicate the source key for each row
all_data = []
if "data" in response_json:
  for key, value in response_json["data"].items():
    if isinstance(value, list):
      for item in value:
        item["source_key"] = key
        all_data.append(item)
    else:
      value["source_key"] = key
      all_data.append(value)
else:
  print("'data' key not found in response.")

if all_data:
  df = pd.DataFrame(all_data)
else:
  print("No data available to create DataFrame.")
  df = pd.DataFrame()  # Empty DataFrame as fallback

# Export to CSV
output_file = "output.csv"
df.to_csv(output_file, index=False)
print("Exported to output.csv")

if "errors" in response_json:
    print("API errors:", response_json["errors"])