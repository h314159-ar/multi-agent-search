import transactions
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
df = pd.DataFrame(te.fit(transactions).transform(transactions),
                  columns=te.columns_)