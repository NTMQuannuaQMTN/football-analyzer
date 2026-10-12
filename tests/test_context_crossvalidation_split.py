from cv.train_context_crossvalidation import partitions


def test_context_folds_hold_out_training_validation_and_test_matches():
    matches=[str(i) for i in range(7)];tests=[]
    for index in range(7):
        train,val,test=partitions(matches,index)
        assert len(train)==5 and len(set(train+[val,test]))==7
        assert val not in train and test not in train and val!=test
        tests.append(test)
    assert tests==matches
