from dotenv import load_dotenv
import os
load_dotenv()

uri  = os.getenv('NEO4J_URI')
user = os.getenv('NEO4J_USERNAME')
pw   = os.getenv('NEO4J_PASSWORD')

print(f'URI: {uri}')

from neo4j import GraphDatabase
try:
    driver = GraphDatabase.driver(uri, auth=(user, pw))
    driver.verify_connectivity()
    with driver.session() as s:
        result = s.run('MATCH (n) RETURN count(n) AS total')
        total = result.single()['total']
    print(f'OK — connected. {total} nodes in graph.')
    driver.close()
except Exception as e:
    print(f'FAIL — {e}')