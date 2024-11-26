from neo4j import GraphDatabase
import os
from .env import Env
import timeit

graphdb_driver = GraphDatabase.driver(uri=Env.neo4j_url, 
                                      auth=(
                                          Env.neo4j_user,
                                          Env.neo4j_password
                                          )
                                        )
