# Author: lqxu

import pandas as pd
from core.utils import data_provider


if __name__ == '__main__':
    test_df = pd.DataFrame({
        "sentence": ["今天天气不错", "明天天气也很好哦"]
    })

    tokenizer = data_provider.get_default_tokenizer()

    new_df = pd.DataFrame(
        {
            key: value for key, value in tokenizer(
                test_df["sentence"].to_list(), return_attention_mask=False, return_token_type_ids=False
            ).items()
        }
    )
    print(new_df)
    print(type(new_df))
