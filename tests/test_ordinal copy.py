import mankey.custom_helpers as transformers
import pandas as pd


def test_basic():
    assert 1 == 1


def test_ordinal_one():
    import pandas as pd

    data = {'Pclass':  ['First_class', 'Second_Class', 'Third_Class', 'Fourth_Class'],
            'level': [1, 2, 3,4],
            }

    df = pd.DataFrame(data)
    levels_dict = {"Pclass": ['First_class', 'Second_Class', 'Third_Class'],
                   }

    target_result = {'type':  [0, 1, 2,3],
                     'level': [1, 2, 3,4],
                     }
    target_df = pd.DataFrame(target_result)

    t_ord = transformers.Ordinal_Transformer()
    t_ord.fit( levels_dict, df,None, input_vars=['Pclass'])

    df = t_ord.transform(df, None)
    pd.testing.assert_frame_equal(df, target_df)
    

