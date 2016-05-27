# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import logging

from rankpy.queries import Queries
from rankpy.queries import find_constant_features

from rankpy.models import LambdaMART


# Turn on logging.
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

# Load the query datasets.
training_queries = Queries.load_from_text('train.svmlight')
validation_queries = Queries.load_from_text('vali.svmlight')
test_queries = Queries.load_from_text('test.svmlight')

logging.info('================================================================================')

# Print basic info about query datasets.
logging.info('Train queries: %s' % training_queries)
logging.info('Valid queries: %s' % validation_queries)
logging.info('Test queries: %s' %test_queries)

logging.info('================================================================================')

# Set this to True in order to remove queries containing all documents
# of the same relevance score -- these are useless for LambdaMART.
remove_useless_queries = False

# Find constant query-document features.
cfs = find_constant_features([training_queries, validation_queries])

# Get rid of constant features and (possibly) remove useless queries.
training_queries.adjust(remove_features=cfs, purge=remove_useless_queries)
validation_queries.adjust(remove_features=cfs, purge=remove_useless_queries)
test_queries.adjust(remove_features=cfs)

# Print basic info about query datasets.
logging.info('Train queries: %s' % training_queries)
logging.info('Valid queries: %s' % validation_queries)
logging.info('Test queries: %s' % test_queries)

logging.info('================================================================================')

model = LambdaMART(metric='nDCG@38', max_leaf_nodes=7, shrinkage=0.1,
                   estopping=4, n_jobs=-1, min_samples_leaf=50,
                   random_state=42)

#TODO: do some crossval here?
model.fit(training_queries, validation_queries=test_queries)

logging.info('================================================================================')
logging.info('%s on the test queries: %.8f'
             % (model.metric, model.evaluate(test_queries, n_jobs=-1)))

model.save('LambdaMART_L7_S0.1_E50_' + model.metric)
predicted_rankings = model.predict_rankings(test_queries)

test_df = pd.read_csv("../test_set_VU_DM_2014.csv", header=0, nrows = test_queries.document_count())
test_df['pred_position'] = np.concatenate(predicted_rankings)
sorted_df = test_df[['srch_id', 'prop_id', 'pred_position']].sort_values(['srch_id', 'pred_position'])

submission = pd.DataFrame({ 'SearchId': sorted_df.srch_id, 'PropertyId': sorted_df.prop_id })[['SearchId', 'PropertyId']]
submission.to_csv('model_%d_%f.csv' % (test_queries.document_count(), model.evaluate(test_queries, n_jobs=-1)), index=False)
