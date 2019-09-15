from pathlib import Path

cross_domain_settings = ['laptops_to_restaurants', 'restaurants_to_laptops']
in_domain_settings = ['laptops14', 'restaurants14', 'restaurants15']
all_settings = cross_domain_settings + in_domain_settings
num_splits = 3
base = str(Path.home()) + '/private-nlp-architect/nlp_architect/models/absa_neural/data/conll/'