import mankey.custom_helpers as transformers
import pandas as pd



def test_basic():
    assert 1 == 1

def test_woe_df():
    X = {"type": ['4wd', '4wd', 'awd', 'awd', '2wd'], "cat": ["toyota", "toyota", "portugal", "portugal", "portugal"], "target":[1, 1, 1, 1, 0]}
    

    df = pd.DataFrame(X)

    target_result = {"type_woe": [13.815511, 13.815511, 13.815511, 13.815511, -13.815511], "cat_woe": [13.815511, 13.815511, 0.693147, 0.693147, 0.693147], }
    target_df = pd.DataFrame(target_result)
    t_woe = transformers.WoE_Transformer()
    t_woe.fit(df, df['target'], 'target', input_vars=['type', 'cat'], num_cat_threshold=1)
    
    t_woe.transform(df,None)
    pd.testing.assert_frame_equal(df,target_df)