import numpy as np
import pandas as pd
from joblib import dump
from scipy import stats
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline


def make_models(data, model):

    conn = sqlite3.connect(f'{data}/{data}_noise.sqlite3')
    cur = conn.cursor()

    test = pd.read_csv(f'{data}/data/{data}_test.csv')

    files = [f'{data}_train',
    f'{data}_train_05',
    f'{data}_lstm_train',
    f'{data}_lstm_train_noise_1',
    f'{data}_lstm_train_noise_15',
    f'{data}_lstm_train_noise_25',
    f'{data}_lstm_train_noise_33',
    f'{data}_lstm_train_noise_5',
    
    ]

    # Logistic Regression
    if model == 'lr':
        pipe_line = make_pipeline(
            TfidfVectorizer(),
            LogisticRegression(solver='lbfgs'))

    # Random Forest
    if model == 'rf':
        pipe_line = make_pipeline(
            TfidfVectorizer(),
            RandomForestClassifier(random_state=0))

    # Passive aggressive classifier
    if model == 'pa':
        pipe_line = make_pipeline(
            TfidfVectorizer(),
            PassiveAggressiveClassifier())


    # Make the SQLite database for results
    create_results_table = f"""
    CREATE TABLE IF NOT EXISTS {data}_{model}_results
    ('model_name', 'train_accuracy', 'test_accuracy', 
    't_stat', 'p_value', 't_stat_05', 'p_value_05', 'average', 'roc_auc', 
    '0_f1', '0_recall', '0_precision', '1_f1', 'f_recall', '1_precision',
    '0_tstat', '0_pvalue', '0_avg', 
    '1_tstat', '1_pvalue', '1_avg',
    '0_tstat_05', '0_pvalue_05', 
    '1_tstat_05', '1_pvalue_05');
    """
    cur.execute(create_results_table)
    conn.commit()


    # Make the models    
    for f in files:    
        if f == f'{data}_train':
            train = pd.read_csv(f'{data}/{data}_data/{data}_train.csv')
            X_train = train['text']
            y_train = train['label']
            X_test = test['text']
            y_test = test['label']
            name = f'{data}_{model}_orig'
            preds = pd.DataFrame({'y_test': y_test})
            
            # Fit the pipeline
            pipeline = pipe_line
            pipeline.fit(X_train, y_train)

            # Pickle that stuff
            dump(pipeline, f'{data}/models/{name}_pipeline.joblib', compress=True)

            # Do predictions
            y_pred = pipeline.predict(X_test)
            preds[name] = y_pred

            # Do accuracy and stuff
            train_accuracy = pipeline.score(X_train, y_train)
            test_accuracy = pipeline.score(X_test, y_test)
            roc_auc = roc_auc_score(y_test, y_pred)
            average = preds[name].mean()
            report = classification_report(y_test, y_pred, output_dict=True)

            # Do t-tests
            orig_preds = preds[name]

            orig_acc = []

            for i in range(0, len(orig_preds)):
                if y_test[i] == y_pred[i]:
                    orig_acc.append(1)
                else:
                    orig_acc.append(0)

            tstats = stats.ttest_ind(orig_acc, orig_acc)

            # T-tests 0
            y_pred_0 = preds[preds['y_test'] == 0]
            avg_0 = y_pred_0[name].mean()
            y_pred_0 = y_pred_0[name]

            orig_preds_0 = preds[preds['y_test'] == 0]
            orig_preds_0 = orig_preds_0[f'{data}_{model}_orig']
            tstats_0 = stats.ttest_ind(y_pred_0, orig_preds_0)
            
            
            # T-tests 1
            y_pred_1 = preds[preds['y_test'] == 1]
            avg_1 = y_pred_1[name].mean()
            y_pred_1 = y_pred_1[name]

            orig_preds_1 = preds[preds['y_test'] == 1]
            orig_preds_1 = orig_preds_1[f'{data}_{model}_orig']

            tstats_1 = stats.ttest_ind(y_pred_1, orig_preds_1)       

            # Put that stuff in the other table
            insert_results = f"""
            INSERT INTO {data}_{model}_results
            ('model_name', 'train_accuracy', 'test_accuracy', 
            't_stat', 'p_value', 'average', 'roc_auc',
            '0_f1', '0_recall', '0_precision',
            '1_f1', 'f_recall', '1_precision',
            '0_tstat', '0_pvalue', '0_avg',
            '1_tstat', '1_pvalue', '1_avg')
            VALUES
            ('{name}', {train_accuracy}, {test_accuracy}, 
            {tstats[0]}, {tstats[1]}, {average}, {roc_auc},
            {report['0']['f1-score']}, {report['0']['recall']}, {report['0']['precision']},
            {report['1']['f1-score']}, {report['1']['recall']}, {report['1']['precision']},
            {tstats_0[0]}, {tstats_0[1]}, {avg_0},
            {tstats_1[0]}, {tstats_1[1]}, {avg_1});"""
            cur.execute(insert_results)
            conn.commit()
            
        
        elif f == f'{data}_train_05':
            train = pd.read_csv(f'{data}/{data}_data/{data}_train_05.csv')
            X_train = train['text']
            y_train = train['label']
            X_test = test['text']
            y_test = test['label']
            name = f'{data}_{model}_orig_05'
            
            # Fit the model
            pipeline = pipe_line
            pipeline.fit(X_train, y_train)

            # Pickle that stuff
            dump(pipeline, f'{data}/models/{name}_pipeline.joblib', compress=True)

            # Do predictions
            y_pred = pipeline.predict(X_test)
            preds[name] = y_pred

            # Do accuracy and stuff
            train_accuracy = pipeline.score(X_train, y_train)
            test_accuracy = pipeline.score(X_test, y_test)
            roc_auc = roc_auc_score(y_test, y_pred)
            average = preds[name].mean()
            report = classification_report(y_test, y_pred, output_dict=True)

            orig_preds = preds[f'{data}_{model}_orig']

            acc_05 = []
            for i in range(0, len(orig_preds)):
                if y_test[i] == y_pred[i]:
                    acc_05.append(1)
                else:
                    acc_05.append(0)

            tstats = stats.ttest_ind(acc_05, orig_acc)
            tstats_05 = stats.ttest_ind(acc_05, acc_05)
            
            # T-tests 0
            y_pred_0 = preds[preds['y_test'] == 0]
            avg_0 = y_pred_0[name].mean()
            y_pred_0 = y_pred_0[name]

            orig_preds_0 = preds[preds['y_test'] == 0]
            orig_preds_0 = orig_preds_0[f'{data}_{model}_orig']

            tstats_0 = stats.ttest_ind(y_pred_0, orig_preds_0)
            
            # T-tests 1
            y_pred_1 = preds[preds['y_test'] == 1]
            avg_1 = y_pred_1[name].mean()
            y_pred_1 = y_pred_1[name]

            orig_preds_1 = preds[preds['y_test'] == 1]
            orig_preds_1 = orig_preds_1[f'{data}_{model}_orig']

            tstats_1 = stats.ttest_ind(y_pred_1, orig_preds_1)
          

            # Put that stuff in the other table
            colls = ['model_name', 'train_accuracy', 'test_accuracy', 
            't_stat', 'p_value', 't_stat_05', 'p_value_05', 
            'average', 'roc_auc',
            '0_f1', '0_recall', '0_precision',
            '1_f1', 'f_recall', '1_precision',
            '0_tstat', '0_pvalue', '0_avg', 
            '1_tstat', '1_pvalue', '1_avg',
            '0_tstat_05', '0_pvalue_05', 
            '1_tstat_05', '1_pvalue_05']
            values = [name, 
            train_accuracy, 
            test_accuracy, 
            tstats[0], 
            tstats[1], 
            tstats_05[0], 
            tstats_05[1], 
            average, 
            roc_auc,
            report['0']['f1-score'], 
            report['0']['recall'], 
            report['0']['precision'],
            report['1']['f1-score'], 
            report['1']['recall'], 
            report['1']['precision'],
            tstats_0[0], 
            tstats_0[1], 
            avg_0, 
            tstats_1[0], 
            tstats_1[1], 
            avg_1, 
            0.0, 
            1.0, 
            0.0, 
            1.0]

            cols = ()
            vals = ()
            for i in range(0, len(colls)):
                if str(values[i]) != 'nan':
                    cols += (colls[i],)
                    vals += (values[i],)
                else:
                    pass
            # Put that stuff in the other table
            insert_results = f"""
            INSERT INTO {data}_{model}_results
            {cols}
            VALUES
            {vals};"""
            cur.execute(insert_results)
            conn.commit()
            

        else:
            if f == f'{data}_lstm_train':
                name = f'{data}_{model}_lstm'
                train = pd.read_csv(f'{data}/{data}_data/{data}_lstm_train.csv')
                train['text'] = train['text'].replace(np.NaN, 'Empty')
                train = train[train['text'] != 'Empty']            
            
            else:
                # Get all set up
                name = f'{f}_{model}'
                train = pd.read_csv(f'{data}/{data}_data/{f}.csv')
                train['text'] = train['text'].replace(np.NaN, 'Empty')
                train = train[train['text'] != 'Empty']
            
            
            X_train = train['text']
            y_train = train['label']
            X_test = test['text']
            y_test = test['label']

            # Do a logistic regression
            pipeline = pipe_line
            pipeline.fit(X_train, y_train)

            # Pickle that stuff
            dump(pipeline, f'{data}/models/{name}_pipeline.joblib', compress=True)

            # Do predictions
            y_pred = pipeline.predict(X_test)            
            preds[name] = y_pred

            # Do accuracy and stuff and save that too
            train_accuracy = pipeline.score(X_train, y_train)
            test_accuracy = pipeline.score(X_test, y_test)
            roc_auc = roc_auc_score(y_test, y_pred)
            average = preds[name].mean()
            report = classification_report(y_test, y_pred, output_dict=True)

            # Do t-tests and save that
            orig_preds = preds[f'{data}_{model}_orig']
            preds_05 = preds[f'{data}_{model}_orig_05']
            
            acc = []
            for i in range(0, len(orig_preds)):
                if y_test[i] == y_pred[i]:
                    acc.append(1)
                else:
                    acc.append(0)

            tstats = stats.ttest_ind(acc, orig_acc)
            tstats_05 = stats.ttest_ind(acc, acc_05)
            
            # T-tests 0
            y_pred_0 = preds[preds['y_test'] == 0]
            avg_0 = y_pred_0[name].mean()
            y_pred_0 = y_pred_0[name]


            orig_preds_0 = preds[preds['y_test'] == 0]
            orig_preds_0 = orig_preds_0[f'{data}_{model}_orig']

            preds_05_0 = preds[preds['y_test'] == 0]
            preds_05_0 = preds_05_0[f'{data}_{model}_orig_05']
            

            tstats_0 = stats.ttest_ind(y_pred_0, orig_preds_0)
            tstats_05_0 = stats.ttest_ind(preds_05_0, y_pred_0)


            # # T-tests 1
            y_pred_1 = preds[preds['y_test'] == 1]
            avg_1 = y_pred_1[name].mean()
            y_pred_1 = y_pred_1[name]

            orig_preds_1 = preds[preds['y_test'] == 1]
            orig_preds_1 = orig_preds_1[f'{data}_{model}_orig']

            preds_05_1 = preds[preds['y_test'] == 1]
            preds_05_1 = preds_05_1[f'{data}_{model}_orig_05']

            tstats_1 = stats.ttest_ind(y_pred_1, orig_preds_1)
            tstats_05_1 = stats.ttest_ind(y_pred_1, preds_05_1)


            colls = ['model_name', 'train_accuracy', 'test_accuracy', 
            't_stat', 'p_value', 't_stat_05', 'p_value_05', 
            'average', 'roc_auc',
            '0_f1', '0_recall', '0_precision',
            '1_f1', 'f_recall', '1_precision',
            '0_tstat', '0_pvalue', '0_avg', 
            '1_tstat', '1_pvalue', '1_avg',
            '0_tstat_05', '0_pvalue_05', 
            '1_tstat_05', '1_pvalue_05']
            values = [name, 
            train_accuracy, 
            test_accuracy, 
            tstats[0], 
            tstats[1], 
            tstats_05[0], 
            tstats_05[1], 
            average, 
            roc_auc,
            report['0']['f1-score'], 
            report['0']['recall'], 
            report['0']['precision'],
            report['1']['f1-score'], 
            report['1']['recall'], 
            report['1']['precision'],
            tstats_0[0], 
            tstats_0[1], 
            avg_0, 
            tstats_1[0], 
            tstats_1[1], 
            avg_1, 
            tstats_05_0[0], 
            tstats_05_0[1], 
            tstats_05_1[0], 
            tstats_05_1[1]]

            cols = ()
            vals = ()
            # for i in range(0, len(colls)):
            #     print(colls[i], values[i])
            # print("____________________________________")
            for i in range(0, len(colls)):
                if str(values[i]) != 'nan':
                    if str(values[i]) != '-inf':
                        cols += (colls[i],)
                        vals += (values[i],)
                else:
                    pass
            # Put that stuff in the other table
            insert_results = f"""
            INSERT INTO {data}_{model}_results
            {cols}
            VALUES
            {vals};"""
            cur.execute(insert_results)
            conn.commit()
    
    # Save predictions
    preds.to_sql(f'{data}_{model}_preds', con=conn, if_exists='replace')


# make_models('enron', 'lr')
# make_models('enron', 'rf')
# make_models('enron', 'pa')

# make_models('imbd', 'lr')
# make_models('imbd', 'rf')
# make_models('imbd', 'pa')

make_models('sms', 'lr')
make_models('sms', 'rf')
make_models('sms', 'pa')