# The Graph Query Examples | Python API with Fastify

Demos how to query a [Graph Network Uniswap V3 Subgraph] published to The Graph Network using an API Key obtained on [The Graph Studio](https://thegraph.com/studio) in a `python` api using the `FastAPI` framework.
This is a handy way to integrate a subgraph into a python api that can be queried from an app without exposing your API Key to that app.
This example could be extended to add authentication, meaning only authenticated users can query the api. As well as by providing custom endpoints that query the subgraph and expose data after processing, etc.

It is a really simple example that accepts a graphql request on the exposed `/graphql` endpoint and routes those requests to the respective Uniswap V3 Subgraph published on The Graph Network.

## Running

```bash
# Clone Repo
git clone git@github.com:graphprotocol/query-examples.git

# CD into nodejs example
cd ./examples/python-fastapi

# Install deps
pip3 install -r requirements.txt

# create env
cp ./.env.example ./.env
# need to set the API_KEY value using an API Key created in subgraph studio

# Query the Uniswap V3 Subgraph with URL
ETH 
https://gateway.thegraph.com/api/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV 

ARB 
https://gateway.thegraph.com/api/subgraphs/id/HyW7A86UEdYVt5b9Lrw8W2F98yKecerHKutZTRbSCX27

POLY 
https://gateway.thegraph.com/api/subgraphs/id/3hCPRGf4z88VC5rsBKU5AA9FBBq5nF3jbKJG7VZCbhjm

# Run
python -m uvicorn main:app --reload

# Run Ethereum query
python csvETH.py

# Run Arbitrum query 
python csvARB.py

# Run Polygon query
python csvPOLY.py

