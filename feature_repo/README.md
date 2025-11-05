Using the data with Feast.

You can look at the Milvus UI via: 

http://127.0.0.1:9091/webui/collections

simply run:

```bash
feast apply
feast materialize --disable-event-timestamp
```

```python
import pandas as pd
from feast import FeatureStore

store = FeatureStore("./")

df = pd.read_parquet("./data/train-00000-of-00157_sample_with_timestamp_chunked.parquet")

y = store.retrieve_online_documents_v2(query=q, features=['wiki_passages:text', 'wiki_passages:embeddings'], top_k=5)
y.to_df()
```

Which will output a sample like so:
```markdown
                                                text                                         embeddings  distance    id
0  Aaron Aaron ( or ; "Ahärôn") is a prophet, hig...  [0.013342111371457577, 0.582173764705658, -0.3...  1.000000   1_1
1  the eastern border-land of Egypt (Goshen). Whe...  [0.013342111371457577, 0.582173764705658, -0.3...  1.000000   1_2
2  included the entire House of Amran. The Quran ...  [0.013342111371457577, 0.582173764705658, -0.3...  0.904321  30_1
3  Book of Exodus, Aaron first functioned as Mose...  [0.013342111371457577, 0.582173764705658, -0.3...  0.899625  30_2
4  the Exodus, in which, according to the Quran a...  [0.013342111371457577, 0.582173764705658, -0.3...  0.904321   2_2
```
