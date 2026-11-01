from neo4j import Driver, GraphDatabase

from app.config import Settings


def connect(settings: Settings) -> Driver:
    if not settings.neo4j_uri:
        raise RuntimeError("NEO4J_URI is not set in .env (see SETUP_ACCOUNTS.md, step 4).")
    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_username, settings.neo4j_password))
    driver.verify_connectivity()
    return driver
