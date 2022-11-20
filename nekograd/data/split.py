from sklearn.model_selection import train_test_split


def train_val_test_split(ids: tuple, qval=0.05, qtest=0.1, random_state=42) -> list:
    tr, test_val = train_test_split(ids, test_size=qval + qtest, random_state=random_state)
    return (tr, *train_test_split(test_val, test_size=qtest / (qval + qtest), random_state=random_state))
