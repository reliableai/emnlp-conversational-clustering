DATASETS = {
    "synthetic": {"path": "data/synthetic_dataset.csv", "item_column": "item"},
    "enterprise_ticket": {
        "path": "data/enterprise_ticket_dataset_1000.csv",
        "item_column": "ticket_text",
        "benchmark_spec_path": "data/enterprise_ticket_dataset.benchmark.json",
    },
    "imdb": {"path": "data/IMDb_movies.csv", "item_column": "title"},
    "pricerunner": {
        "path": "data/pricerunner_aggregate.csv",
        "item_column": "Product Title",
    },
    "us_cities": {"path": "data/us_city_dataset.csv", "item_column": "city_name"},
}
