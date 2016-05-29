import time
import pandas as pd

if __name__ == "__main__":
    ranklib_df = pd.read_csv("model_ranklib_4885.scores", sep="\t", header=None, names=['srch_id', 'local_prod_id', 'score'])
    test_df = pd.read_csv("../test_set_VU_DM_2014.csv", header=0, usecols=['srch_id', 'prop_id'])
    test_df['score'] = -ranklib_df['score']
    sorted_df = test_df[['srch_id', 'prop_id', 'score']].sort_values(['srch_id', 'score'])
    sorted_df.score = -sorted_df.score

    submission = pd.DataFrame({ 'SearchId': sorted_df.srch_id, 'PropertyId': sorted_df.prop_id})[['SearchId', 'PropertyId']]
    submission.to_csv('model_scoring_4965.csv', index=False)
