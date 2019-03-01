import pandas as pd
import json


def get_questionnaire_dataframe(worker_ids, questionnaires):
    df_list = []

    for idx, quest in zip(worker_ids, questionnaires):
        tmp = json.loads(quest)
        tmp["WorkerId"] = idx

        df_list.append(tmp)

    df = pd.DataFrame(df_list)
    df = df.set_index("WorkerId")
    df = df[~df.index.duplicated(keep='first')]

    return df


def get_sum_true_false(df, columns):
    return df.apply(lambda x: x[columns].values.sum(), axis=1)
