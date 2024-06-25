import requests


class SearchEngine:
    BASE_URL = "http://3.36.208.188:8989"
    ALLOWED_TYPES = {"self", "ensemble", "dense", "sparse"}

    def __init__(self, type, endpoint):
        if type not in self.ALLOWED_TYPES:
            raise TypeError(f"Invalid type '{type}'. Allowed types are: {', '.join(self.ALLOWED_TYPES)}")
        self.type = type
        self.endpoint = self.BASE_URL + endpoint

    def search(self, query, api_key, score_threshold, top_k, index):
        payload = {
            "input": query,
            "openai_key": api_key,
            "score_threshold": score_threshold,
            "top_k": top_k,
            "workspace_id": index
        }
        try:
            response = requests.post(self.endpoint, json=payload)
            return response.json()
        except requests.RequestException as e:
            print(f"An error occurred while making the request: {e}")
            return None


class SearchManager:
    def __init__(
            self,
            api_key: str,
            index: str,
            top_k: float,
            score_threshold: float

    ):
        self.engines = {}
        self.api_key = api_key
        self.index = index
        self.top_k = top_k
        self.score_threshold = score_threshold

    def add_engine(self, type: str):
        self.endpoint = self.get_endpoints(type)
        self.engines[type] = SearchEngine(type, self.endpoint)

    def get_endpoints(self, type: str) -> str:
        endpoint = "/api/v1/retrieval/self_query/"
        if type == "dense":
            endpoint = "/api/v1/retrieval/self_query/"
        elif type == "sparse":
            endpoint = "/api/v1/retrieval/self_query/"
        elif type == "ensemble":
            endpoint = "/api/v1/retrieval/self_query/"
        return endpoint

    def search_all(
            self,
            query: str,
    ):
        results = {}
        for name, engine in self.engines.items():
            results[name] = engine.search(
                query,
                api_key=self.api_key,
                score_threshold=self.score_threshold,
                top_k=self.top_k,
                index=self.index
            )
        return results


# Usage example
if __name__ == "__main__":
    search_manager = SearchManager(
        api_key="",
        index="86f92d0e-e8ec-459a-abb8-0262bbf794a2",
        top_k=5,
        score_threshold=0.7
    )

    # Add search engines
    search_manager.add_engine("self")

    # Perform search on all engines
    query = "봉준호 감독이 만든 영화 추천좀?"
    results = search_manager.search_all(query)

    # Print results
    for engine_name, result in results.items():
        print(f"Results from {engine_name}:")
        print(result)

