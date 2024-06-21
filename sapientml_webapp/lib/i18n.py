# Copyright 2023-2024 The SapientML Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from copy import deepcopy
from typing import Literal

# internationalization
I18N_DICT = {
    "header": {
        "welcome": {
            "en": "**NOTE:** Welcome to SapientML. This is the tutorial.",
            "ja": "**NOTE:** SapientMLにようこそ。これはチュートリアルです。",
        },
        "training_data": {
            "en": "Training Data",
            "ja": "学習データ",
        },
        "configuration": {
            "en": "Configurations",
            "ja": "機械学習の要件",
        },
        "execution": {
            "en": "Code generation logs",
            "ja": "コード生成ログ",
        },
        "result": {
            "en": "Result",
            "ja": "コード生成結果",
        },
        "button_display_configuration": {
            "en": "Display configuration",
            "ja": "実行時パラメータを表示する",
        },
        "experimental_result": {
            "en": "Result",
            "ja": "実行結果",
        },
        "model_details": {
            "en": "Model Details",
            "ja": "モデル詳細",
        },
        "correlation_feature_and_target": {
            "en": "Correlation between feature and target column",
            "ja": "目的変数と特徴量の関係",
        },
        "prediction": {
            "en": "Prediction by model built with generated code",
            "ja": "生成コードで構築したモデルによる予測の実行",
        },
        "prediction_result": {
            "en": "Prediction Result",
            "ja": "予測結果",
        },
        "metrics": {
            "en": "Prediction Score",
            "ja": "メトリクスの表示",
        },
    },
    "sidebar": {
        "to_trial": {
            "en": "Go to App",
            "ja": "アプリへ進む",
        },
        "to_tutorial": {
            "en": "Back to the tutorial",
            "ja": "チュートリアルに戻る",
        },
    },
    "welcome": {
        "description": {
            "en": "**NOTE:** SapientML generates Python code that builds a machine learning model \
                that meets the requirements of a machine learning task for the uploaded CSV-formatted tabular data file. \
                The generated Python code includes code for data preprocessing, training and testing the model, \
                and exploratory data analysis (EDA).  The code can be downloaded as Python source code files (.py) \
                and Jupyter notebooks (.ipynb).  Please note that the code performing EDA is only included in the notebook.",
            "ja": "**NOTE:** SapientMLは、アップロードされたCSV形式の表データに対して、機械学習タスクの要件を満たす\
                機械学習モデルを構築するためのPythonコードを生成します。生成されたPythonコードには、データの前処理、モデル構築、\
                および探索的データ解析（EDA）のコードが含まれています。コードは、Python（.py）形式とJupyter Notebook（.ipynb）形式の\
                両方でダウンロードできます。ただし、EDA部分のコードはノートブックにのみ含まれていることに注意してください。",
        },
    },
    "upload_form": {
        "select_file": {
            "en": "Choose a dataset (.csv) for generating ML code",
            "ja": "機械学習コードを生成したいデータセット (.csv)を選択してください",
        },
        "column_name_rule": {
            "en": "Please use dataset (.csv) without \\\\ in column names",
            "ja": "カラム名に \\\\ を含まないデータセット (.csv)をご利用ください",
        },
        "encoding": {
            "en": "Encoding",
            "ja": "文字コード",
        },
        "btn_submit": {
            "en": "Submit",
            "ja": "確定",
        },
        "btn_sample": {
            "en": "Use sample dataset",
            "ja": "サンプルデータセットを利用する",
        },
        "sample_information": {
            "en": "{source} is used as the sample dataset.",
            "ja": "{source}をサンプルデータセットとして使用しています。",
        },
        "no_file_error": {
            "en": "Please select a file.",
            "ja": "ファイルを選択してください。",
        },
        "description": {
            "en": "Please upload the table data (CSV format) and press the {button} button.",
            "ja": "表データ（CSV形式）をアップロードして、{button}ボタンを押してください。",
        },
        "tutorial": {
            "en": "**NOTE:** In the tutorial, you cannot upload data. Please press the {button} button. \
                Afterwards the {source} will be loaded as the sample dataset.",
            "ja": "**NOTE:** チュートリアルでは表データのアップロードはできません。\
                {button}ボタンを押すとサンプルデータとして{source}が読み込まれます。",
        },
    },
    "training_data": {
        "data_size": {
            "en": "Shape of data: {columns} cols x {rows} rows. Displaying top 50 rows.",
            "ja": "データサイズ： {columns} cols × {rows} rows。先頭の50行を表示しています。",
        },
        "button_display_training_data": {
            "en": "Display training data",
            "ja": "学習データを表示する",
        },
        "illigal_column_name_error": {
            "en": "Failed to upload csv file. Use csv without \\\\ in column name.",
            "ja": "csvファイルのアップロードに失敗しました。カラム名に \\\\ を含まないcsvを利用してください。",
        },
        "description": {
            "en": "",
            "ja": "",
        },
    },
    "configuration": {
        "task_type": {
            "en": "Task Type :red[*required]",
            "ja": "タスクの種類 :red[*必須]",
        },
        "task_type_desc": {
            "en": ":red[(required)]Select the type of machine learning task: classification or regression. \
                By default, the estimated value is calculated from the target columns.",
            "ja": ":red[（必須）] 機械学習タスクの種類として classification (分類)、 regression (回帰)のいずれかを選択します。\
                デフォルトの設定は目的変数から推測したタスクの種類です。",
        },
        "target_column_names": {
            "en": "Target Columns :red[*required]",
            "ja": "目的変数 :red[*必須]",
        },
        "target_column_names_desc": {
            "en": ":red[(required)]Select the column name you want the machine learning task to predict.",
            "ja": ":red[（必須）] 機械学習タスクで予測させたいカラム名を選択します。",
        },
        "target_column_names_error": {
            "en": "Please select at least 1 Target Column Names",
            "ja": "目的変数を少なくとも1個選択してください",
        },
        "advanced_settings": {
            "en": "Advanced Settings",
            "ja": "詳細設定",
        },
        "advanced_settings_tutorial": {
            "en": "Advanced Settings(You cannot change the settings in the tutorial.)",
            "ja": "詳細設定（チュートリアルでは設定の変更はできません。）",
        },
        "task_settings": {"en": "Task Settings", "ja": "タスク設定"},
        "ignore_columns": {
            "en": "Ignore Columns",
            "ja": "学習に利用しないカラム",
        },
        "ignore_columns_desc": {
            "en": "Select the column names you want to ignore when learning the machine learning model. \
                For example, a variable that has an extreme correlation with an ID column or \
                a target variable column.",
            "ja": "機械学習モデルの学習時に無視させたいカラム名を選択します。例えばID列や目的変数と極端に相関のある変数などです​。",
        },
        "adaptation_metric": {
            "en": "Adaptation Metric",
            "ja": "評価指標",
        },
        "adaptation_metric_desc": {
            "en": "Select a measure for your machine learning model. Available metrics are \
                [F1, AUC, Accuracy, Gini, LogLoss, ROC_AUC, MCC] for classification and [R2, RMSLE, RMSE, MAE] \
                for regression.",
            "ja": "機械学習モデルの評価指標を選択します。選択可能な評価指標はclassificationの場合は \
                [F1, AUC, Accuracy, Gini, LogLoss, ROC_AUC, MCC​]、regressionの場合は [R2, RMSLE, RMSE, MAE]です。​",
        },
        "timeout": {"en": "Timeout (seconds)", "ja": "タイムアウト (秒)"},
        "timeout_desc": {
            "en": "SapientML generates Python code for each of the three candidate machine learning models \
                based on the features of table data and executes training and evaluation with a small amount of data. \
                Then, based on the adaptation metrics, it generates the final code for the machine learning model \
                with the best score. Here, a timeout is set for each code execution.",
            "ja": "SapientMLは、候補となる3つの機械学習モデルのそれぞれに対して、Pythonコードを作成して\
                少量のデータで学習と評価を行います。評価指標に基づき最も優れたスコアの機械学習モデルに対して、\
                最終的なコードを生成します。ここでは各コードの1回の実行に対するタイムアウトを指定してします。",
        },
        "train_data_split": {"en": "Train Data Split", "ja": "学習データ分割"},
        "train_data_split_caption": {
            "en": 'Split the train data to prepare validation data for evaluating the performance of \
                the learning model. The default is "random" to split randomly. You can also select "group", \
                which uses GroupShuffleSplit.',
            "ja": '学習済モデルの性能評価用データを準備するために学習データを分割します。分割方法のデフォルトは\
                "random" (ランダムに分割)です。"group" (GroupShuffleSplitで分割)、"time" (TimeSerisSplitで分割)も指定できます。',
        },
        "split_method": {
            "en": "Split Method",
            "ja": "データ分割方法",
        },
        "split_column_name": {"en": "Column Name for Split", "ja": "カラム名"},
        "seed": {
            "en": "Seed",
            "ja": "ランダムシード",
        },
        "split_train_size": {
            "en": "Train Split Percentage (%)",
            "ja": "データ分割時の学習データの割合",
        },
        "split_train_size_desc": {
            "en": "SapientML internally divides table data into training data and test data. Here, \
                the ratio of training data is set.",
            "ja": "SapientMLは表データを内部で学習データとテストデータに分割します。\
                ここでは学習データの割合を指定します。",
        },
        "hyperparameterTuning": {
            "en": "Hyperparameter Tuning",
            "ja": "ハイパーパラメータ探索",
        },
        "hyperparameterTuning_desc": {
            "en": 'When you check "Enable", the generated code will include code \
                for executing hyperparameter tuning of the machine learning model. \
                If you execute hyperparameter tuning, you also set the "The number of Trials" and "Seed".',
            "ja": "「有効にする」をチェックすると、生成コードに機械学習モデルのハイパーパラメータ探索を行うコードが追加されます。\
                ハイパーパラメータ探索を行う場合は「試行回数」と「ランダムシード」も設定します。​",
        },
        "enable": {
            "en": "Enable",
            "ja": "有効にする",
        },
        "hyperparameterTuningNTrials": {
            "en": "The number of Trials",
            "ja": "試行回数",
        },
        "hyperparameterTuningTimeout": {
            "en": "Timeout",
            "ja": "タイムアウト",
        },
        "value": {
            "en": "Value",
            "ja": "設定値",
        },
        "hpo_seed": {
            "en": "Seed",
            "ja": "ランダムシード",
        },
        "button_display_configuration": {
            "en": "Display configuration",
            "ja": "実行時パラメータを表示する",
        },
        "explanations": {"en": "**Explanations**", "ja": "**データの説明**"},
        "explanations_desc": {
            "en": "To include the code for calculating Permutation Feature Importance in the generated code. \
                This helps in understanding the model better.",
            "ja": "Permutation Feature Importanceを計算するコードを生成コードに含めます。これはモデルの理解に役立ちます。",
        },
        "enable_eda": {"en": "Enable exploratory data analysis (EDA)", "ja": "探索的データ解析 (EDA)を有効にする"},
        "show_charts": {"en": "Show charts about the dataset", "ja": "データセットについてのチャートを表示する"},
        "permutation_importance": {
            "en": "Calculate permutation feature importance",
            "ja": "Permutation Feature Importanceを計算する",
        },
        "permutation_importance_decs": {
            "en": "If the number of columns in the dataset exceeds 100, it takes a huge amount of time, \
                so even if you check it, the calculation will be skipped. \
                If the number of columns increases to more than 100 as a result of preprocessing, \
                they are also skipped.",
            "ja": "表データのカラム数が100を超えると膨大な時間を要するようになるため、チェックしていても計算がスキップされます。\
                前処理の結果、カラム数が増えて100を超えた場合もスキップされます。",
        },
    },
    "codegen": {
        "start_generation": {"en": "Start Code Generation", "ja": "コード生成を開始"},
        "generating_code": {
            "en": "Generating Code...",
            "ja": "コードを生成しています...",
        },
        "failed": {
            "en": "Failed",
            "ja": "エラーが発生しました",
        },
        "canceled": {
            "en": "Canceled by User",
            "ja": "ユーザによりキャンセルされました",
        },
        "result": {
            "en": "Preparing Code Generation Result...",
            "ja": "コード生成結果を処理しています...",
        },
    },
    "experimental_result": {
        "selected_preprocess": {
            "en": "Selected Preprocess",
            "ja": "選択された前処理",
        },
        "heading_candidates": {
            "en": "Candidate models and metrics in validation",
            "ja": "候補モデルと検証時のスコア",
        },
        "description_candidates": {
            "en": "In validation, the model which has the best score is selected.",
            "ja": "最も良いスコアの候補モデルが選択されます。",
        },
        "execution_time": {
            "en": "Execution time",
            "ja": "実行時間",
        },
        "execution_time_body": {
            "en": "{hour} hours {minute} minutes {second} seconds",
            "ja": "{hour}時間{minute}分{second}秒",
        },
        "button_display_logs": {
            "en": "Display code generation logs",
            "ja": "コード生成ログを表示する",
        },
        "script": {"en": "Generated code achieving the best score", "ja": "最良スコアの生成コード"},
        "script_ipynb": {
            "en": "Generated code(Jupyter Notebook format) achieving the best score and data visualization",
            "ja": "最良スコアの生成コード（Jupyter Notebook形式）とデータの可視化",
        },
        "download": {
            "en": "Download generated code",
            "ja": "生成コードをダウンロードする",
        },
        "description": {
            "en": "Here you can see the preprocessing and its order that SapientML selected, \
                and the scores for the three machine learning models that it validated. \
                You can also download the generated code.",
            "ja": "ここではSapientMLが選択した前処理とその順序、検証した3つの機械学習モデルのスコアを確認できます。\
                また生成された各種コードをダウンロードすることができます。",
        },
        "tutorial": {
            "en": "**NOTE:** You cannot download the generated code during the tutorial.",
            "ja": "**NOTE:** チュートリアルでは生成されたコードのダウンロードはできません。",
        },
    },
    "model_details": {
        "feature_importance": {
            "en": "Feature Importance",
            "ja": "特徴量の重要度",
        },
        "warning_no_pi_calculation": {
            "en": "Permutation Importance(PI) is not calculated.",
            "ja": "Permutation Importance(PI)の計算が実行されていません。",
        },
        "hint_to_calculate_pi": {
            "en": "By deleting text columns ({text_columns}) or decreasing number of columns, \
                PI may be shown after generating code again.",
            "ja": "テキストカラム（{text_columns}）を削除、または、カラム数を削減し、コード生成することで、PIが表示される可能性があります。",
        },
        "hint_to_calculate_pi_with_no_text_columns": {
            "en": "By decreasing number of columns to {limit_PI} or less, \
                PI may be shown after generating code again.",
            "ja": "カラム数を{limit_PI}以下に削減し、コード生成することで、PIが表示される可能性があります。",
        },
        "description": {
            "en": "Here you can see the importance of the features to the model.",
            "ja": "ここではモデルに対する特徴量の重要度を見ることができます。",
        },
        "description_word": {
            "en": "SapientML converts text to words and graphs them word by word. \
                All numbers in the text are converted to 'num'.",
            "ja": "SapientMLではテキストカラムは、テキストを単語に変換し、単語ごとにグラフ化しています。\
                テキスト内のすべての数値は'num'に変換しています。",
        },
    },
    "correlation_feature_and_target": {
        "label_selectbox_for_target": {
            "en": "Select column to show the correlation.",
            "ja": "どの目的変数の関係を確認しますか?",
        },
        "label_selectbox_for_feature": {
            "en": "Select feature to show the correlation.",
            "ja": "どの特徴量の関係を確認しますか?",
        },
        "title_graph_scatter": {
            "en": "Scatterplot of target and feature",
            "ja": "特徴量と目的変数の散布図",
        },
        "title_graph_box": {
            "en": "Boxplots of target and feature",
            "ja": "特徴量と目的変数の箱ひげ図",
        },
        "title_graph_bar": {
            "en": "Stacked bar chart of target and feature",
            "ja": "特徴量と目的変数の積み上げ棒グラフ",
        },
        "description_graph_text": {
            "en": "Text is converted to words and graphed word by word. All numbers in text were converted to 'num'.",
            "ja": "テキストデータを単語に変換し、単語ごとにグラフ化しています。テキスト内のすべての数値は'num'に変換しています。",
        },
        "description": {
            "en": "To make it easier to understand the data and model, we visualize the relationship \
                between the Target Columns and the features. \
                By understanding the relationship between the Target Columns and the features, \
                we can grasp which features affect the Target Columns. Also, \
                by visualizing the distribution of the features, we can check for data bias and outliers.",
            "ja": "データとモデルを理解しやすくするために目的変数と特徴量の関係を可視化しています。\
                目的変数と特徴量の関係からどの特徴量が目的変数に影響を与えるか把握することができます。\
                また特徴量の分布を可視化することでデータの偏りや外れ値の有無を確認することができます。",
        },
        "description_eda": {
            "en": "The generated code in Jupyter Notebook format includes code for data visualization, \
                such as exploratory data analysis (EDA), and its execution results. Please use them together.",
            "ja": "Jupyter Notebook形式の生成コードには探索的データ解析 (EDA)などのデータ可視化用のコードと\
                その実行結果が含まれていますので併せてご利用下さい。",
        },
    },
    "prediction": {
        "training_model": {"en": "There is no model. Wait for training the model...", "ja": "モデルを構築しています..."},
        "predicting": {
            "en": "Predicting...",
            "ja": "予測しています...",
        },
        "label_file_uploader": {
            "en": "Upload .csv file to be used on prediction. Encoding must be the same as that of the training data.",
            "ja": "予測するデータのファイル(.csv)をアップロードしてください。文字コードは学習データと同じにしてください。",
        },
        "label_character_code_selector": {
            "en": "Select the character code of the file to upload",
            "ja": "アップロードするファイルの文字コードを選択してください",
        },
        "character_code_error": {
            "en": "Set the character code correctly",
            "ja": "文字コードを正しく設定してください",
        },
        "data_size": {
            "en": "Shape of data: {columns} cols x {rows} rows. Displaying top 50 rows.",
            "ja": "データサイズ： {columns} cols × {rows} rows。先頭の50行を表示しています。",
        },
        "btn_predict": {
            "en": "Predict",
            "ja": "予測",
        },
        "data_error_in_number_of_columns": {
            "en": "The uploaded data differs from the training data in column names or the number of columns. \
                Please check the data.",
            "ja": "学習データとカラム名またはカラムの数が異なります。データを確認してください。",
        },
        "data_error_column_order": {
            "en": "The uploaded data differs from the training data in the order of columns. Please, \
                check the data.",
            "ja": "学習データとカラムの順序が違います。データを確認してください。",
        },
        "data_error_display_details": {
            "en": "Display details",
            "ja": "詳細を表示する",
        },
        "data_error_exit_only_train_data": {
            "en": "Columns that exist only in the Train data : {columns}",
            "ja": "学習データのみに存在するカラム : {columns} ",
        },
        "data_error_exit_only_upload_data": {
            "en": "Columns that exist only in Upload Data : {columns}",
            "ja": "アップロードしたデータのみに存在するカラム : {columns}",
        },
        "label_input_grid_height": {
            "en": "Grid height (Adjusting height of rendered size)",
            "ja": "Grid height(データ領域の高さを変更できます)",
        },
        "message_for_editing": {
            "en": "After editing input data above if necessary, you can execute prediction \
                by pushing [Predict] button.",
            "ja": "必要があれば、上の表でデータ編集をして[Predict]ボタンをクリックしてください。",
        },
        "description": {
            "en": "You can use the generated code to train machine learning models and execute classification \
                or prediction tasks. Upload the table data that you want to classify or predict, \
                and then press :green[**Predict**]. If the table data contains Target Columns, \
                you can view the metrics.",
            "ja": "生成コードでモデル構築と予測を実行することができます。予測するデータを学習データと同様にアップロードして\
                :green[**予測**]を押してください。予測するデータに目的変数が含まれていた場合は、メトリクスを見ることができます。",
        },
        "tutorial": {
            "en": "**NOTE:** In the tutorial, you cannot upload data and are allowed to use the sample dataset \
                by pressing the :green[**Submit**].",
            "ja": "**NOTE:** チュートリアルでは表データのアップロードはできません。:green[**確定**]を押すことでサンプルデータを使用できます。",
        },
        "tutorial_pred": {
            "en": "**NOTE:** Please press :green[**Predict**] button to predict Target Columns.",
            "ja": "**NOTE:** :green[**予測**]ボタンを押して予測を開始してください。",
        },
    },
    "prediction_result": {
        "data": {"en": "Displaying top 50 rows", "ja": "先頭の50行を表示しています"},
        "download_result_csv": {
            "en": "Download Result: {download_link}",
            "ja": "予測結果のダウンロード: {download_link}",
        },
    },
    "inquiries": {
        "description": {
            "en": "We will continue to improve and add features to SapientML, \
                so please look forward to the future of SapientML.",
            "ja": "私たちは今後も継続してSapientMLの性能改善・機能追加をしていきますので、SapientMLの今後にご期待ください。",
        },
        "copyright": {
            "en": "&copy; 2023, The SapientML Authors. All rights reserved.",
            "ja": "&copy; 2023, The SapientML Authors. All rights reserved.",
        },
        "tutorial": {
            "en": "**NOTE:** The tutorial is now complete. How was it? \
                Please try the [App](./?page=app&lang=en) with your own data.",
            "ja": "**NOTE:** チュートリアルはここまでです。いかがでしたか？ ぜひ、[アプリ](./?page=app&lang=ja)にて\
                お手持ちのデータでもお試しください。",
        },
    },
    "preprocess": {
        "PREPROCESS:MissingValues:fillna:pandas": {
            "en": "Replacing missing values",
            "ja": "欠損値補完",
        },
        "PREPROCESS:Category:LabelEncoder:sklearn": {
            "en": "Categorical data encoding (OrdinalEncoder)",
            "ja": "カテゴリ変数変換 (数値変換)",
        },
        "PREPROCESS:Category:get_dummies:pandas": {
            "en": "Categorical data encoding (OneHotEncoder)",
            "ja": "カテゴリ変数変換 (one-hot変換)",
        },
        "PREPROCESS:Text:TfidfVectorizer:sklearn": {
            "en": "Transform text",
            "ja": "テキスト変換",
        },
        "PREPROCESS:Scaling:STANDARD:sklearn": {
            "en": "Standardization",
            "ja": "正規化",
        },
        "PREPROCESS:GenerateColumn:DATE:pandas": {
            "en": "Date preprocessing",
            "ja": "日付前処理",
        },
        "PREPROCESS:TextProcessing:Processing:custom": {
            "en": "Text preprocessing",
            "ja": "テキスト前処理",
        },
        "PREPROCESS:Balancing:SMOTE:imblearn": {
            "en": "Oversampling",
            "ja": "オーバーサンプリング",
        },
        "PREPROCESS:Scaling:log:custom": {
            "en": "Logarithmic transformation",
            "ja": "対数変換",
        },
        "PREPROCESS:none": {
            "en": "Nothing",
            "ja": "なし",
        },
    },
    "error": {
        "header": {
            "en": "SapientML didn't complete successfully.",
            "ja": "SapientMLは正常に終了しませんでした。",
        },
        "status": {
            "en": "SapientML Status",
            "ja": "SapientML Status",
        },
        "logs": {
            "en": "Execution Logs",
            "ja": "実行ログ",
        },
        "general": {"en": "Error occured.", "ja": "エラーが発生しました。"},
    },
}


class I18n:
    def __init__(self, i18n_dict, locale, fallback_locale):
        self.i18n_dict = i18n_dict
        self.fallback_locale = fallback_locale
        self.locale = locale

    def t(self, query: str, locale=None):
        if locale is None:
            locale = self.locale
        result = self.gettext(query, locale)
        if result is None:
            result = self.gettext(query, self.fallback_locale)
        if result is None:
            result = query
        return result

    def gettext(self, query, locale):
        parts = query.split(".") + [locale]

        cursor = self.i18n_dict
        for query_part in parts:
            if query_part not in cursor.keys():
                return None
            cursor = cursor[query_part]

        if not isinstance(cursor, str):
            return None
        return cursor


def use_translation(lang: Literal["ja", "en"], I18N_DICT):
    _i18n = I18n(I18N_DICT, lang, "en")
    return _i18n.t


def get_dict_tabular(sampledata=None):
    I18N_DICT_FOR_TABULAR = deepcopy(I18N_DICT)
    I18N_DICT_FOR_TABULAR.update(
        {
            "sample_data": {
                "sample_select": {
                    "en": "**NOTE:** Please select a sample dataset from the dropdown menu.",
                    "ja": "**NOTE:** チュートリアルでは選択したサンプルデータでSapientMLを実行することができます。",
                },
                "stb_label": {
                    "en": "Please select the dataset to use.",
                    "ja": "利用するデータセットを選択してください",
                },
                "titanic_overview": {
                    "en": "Titanic Dataset: Prediction of Survival of Crew and Passengers in the Titanic Disaster",
                    "ja": "Titanic Dataset: タイタニック沈没事故における乗員乗客の生死予測",
                },
                "titanic_details": {
                    "en": "**NOTE:** {source} includes passenger and crew information about the 1912 sinking of the Titanic. \
                        This tutorial predicts column **survived** indicating whether each passenger or crew was survived \
                        (0: Dead; 1: Survived).",
                    "ja": "**NOTE:** {source}は、1912年に発生したタイタニック号の沈没事故に関する乗客と乗組員の情報を含んでいます。\
                        このチュートリアルでは、各乗客や乗組員が生還したかどうかを示す **survived** （0：死亡、1：生存）の推論を行います。",
                },
                "hotel_overview": {
                    "en": "Hotel Cancellation: Prediction of Hotel Reservation Cancellations",
                    "ja": "Hotel Cancellation: ホテルにおける宿泊予約のキャンセル予測",
                },
                "hotel_details": {
                    "en": "**NOTE:** {source} includes information about hotel reservations, including details about the \
                        reservation holders and booking information. In this tutorial, we will perform inference on the \
                        **Status** column(S: Stay, C: Cancellation) to predict \
                            whether the reservation was ultimately stayed or canceled.",
                    "ja": "**NOTE:** {source}はホテルの予約に関する予約者や予約内容の情報を含んでいます。このチュートリアルでは、\
                        最終的に宿泊したかどうかを示す **Status** （S:宿泊、C：キャンセル）の推論を行います",
                },
                "housing_overview": {
                    "en": "Housing Prices: Prediction of Housing Prices",
                    "ja": "Housing Prices: 住宅価格の予測",
                },
                "housing_details": {
                    "en": "**NOTE:** The {source} contains information that influences housing prices, \
                        such as specifications like the number of rooms and location. In this tutorial, \
                        we will perform inference on the **'SalePrice' column** to predict the sales price of the houses.",
                    "ja": "**NOTE:** {source}は部屋の数などの仕様や立地など住宅価格に影響を与える情報を含んでいます。\
                        このチュートリアルでは、住宅の販売価格を示す **SalePrice** の推論を行います",
                },
                "medical_overview": {
                    "en": "Medical Insurance Charges: Medical Insurance Charges Prediction",
                    "ja": "Medical Insurance Charges: 医療保険料の予測",
                },
                "medical_details": {
                    "en": "**NOTE:** {source} includes information that influences medical expenses, \
                        such as the age, gender, and BMI of the policyholders. In this tutorial, \
                        we will perform inference on the **'charges' column** to predict the insurance premiums.",
                    "ja": "**NOTE:** {source}は契約者の年齢、性別BMIなど医療費に影響を与える情報を含んでいます。このチュートリアルでは、\
                        保険料を示す **charges** の推論を行います",
                },
            }
        }
    )
    I18N_DICT_FOR_TABULAR["configuration"].update(
        {
            "task_type_tutorial": {
                "en": "**NOTE:** Please set **{task_type}** for the Task Type in the tutorial",
                "ja": "**NOTE:** チュートリアルではタスクの種類を **{task_type}** に設定してください",
            },
            "target_columns_tutorial": {
                "en": "**NOTE:** Please set **{target_columns}** for the Target Column Names in the tutorial",
                "ja": "**NOTE:** チュートリアルでは目的変数を **{target_columns}** に設定してください",
            },
            "tutorial": {
                "en": "**NOTE:** In the tutorial, please select a single column **{target_columns}** for Target Column Names \
                        and **{task_type}** for Task Type. After that, please press {button} button.",
                "ja": "**NOTE:** チュートリアルでは目的変数に **{target_columns}** 、タスクの種類に **{task_type}** を選択してください。\
                        選択後は表示される{button}を押してください。",
            },
        },
    )

    if sampledata == "Titanic Dataset":
        I18N_DICT_FOR_TABULAR["training_data"].update(
            {
                "tutorial": {
                    "en": "**NOTE:** {source} includes passenger and crew information about the 1912 sinking of the Titanic. \
                        This tutorial predicts column **survived** indicating whether each passenger or crew was survived \
                        (0: Dead; 1: Survived).",
                    "ja": "**NOTE:** {source}は、1912年に発生したタイタニック号の沈没事故に関する乗客と乗組員の情報を含んでいます。\
                        このチュートリアルでは、各乗客や乗組員が生還したかどうかを示す **survived** （0：死亡、1：生存）の推論を行います。",
                },
            }
        )
        I18N_DICT_FOR_TABULAR["model_details"].update(
            {
                "tutorial": {
                    "en": "**NOTE:** From the Permutation Feature Importance graph of the sample data, \
                        we can see that gender (**sex**), **fare**, **age** and cabin grade (**Pclass**) have an impact \
                        on life and death. You can also see that the text column, **ticket**, is decomposed. For example, \
                        **PC 17613** in column **ticket** is decomposed into **PC** and **17613**, \
                        and the number is replaced with **num**. \
                        It is represented by the series **ticket_pc** and **ticket_num** in the chart.",
                    "ja": "**NOTE:** サンプルデータのPermutation Feature Importanceのグラフからは、性別（**sex**）や運賃（**fare**）、\
                        年齢（**age**）、客室の等級（**Pclass**）が生死（**survived**）に影響を与えていることがわかります。\
                        またテキストカラムである**ticket**が分解されていることもわかります。\
                        例えば**ticket**の**PC 17613**は**PC**と**17613**に分解されたうえで数字は**num**に置き換えられます。\
                        グラフの系列として**ticket_pc**と**ticket_num**で表現されます。",
                },
            }
        )
        I18N_DICT_FOR_TABULAR["correlation_feature_and_target"].update(
            {
                "tutorial": {
                    "en": "**NOTE:** Let's see how feature **sex** affects the **survived**. \
                        It is immediately apparent that there are more female survivors.",
                    "ja": "**NOTE:** 性別（**sex**）がどのように目的変数に影響をあたえているか見てみましょう。\
                        女性の生存者が多いことが一目でわかります。",
                },
            }
        )

    elif sampledata == "Hotel Cancellation":
        I18N_DICT_FOR_TABULAR["training_data"].update(
            {
                "tutorial": {
                    "en": "**NOTE:** {source} includes information about hotel reservations, including details about the \
                        reservation holders and booking information. This tutorial predict \
                        **Status** (S: Stay, C: Cancellation) indicating \
                        whether the reservation was ultimately stayed or canceled.",
                    "ja": "**NOTE:** {source}はホテルの予約に関する予約者や予約内容の情報を含んでいます。このチュートリアルでは、\
                        最終的に宿泊したかどうかを示す **Status** （S:宿泊、C：キャンセル）の推論を行います",
                },
            }
        )

        I18N_DICT_FOR_TABULAR["model_details"].update(
            {
                "tutorial": {
                    "en": "**NOTE:** From the Permutation Feature Importance graph of the sample data, \
                        we can see that **Charge** and **Age** have an impact on **Status**.",
                    "ja": "**NOTE:** サンプルデータのPermutation Feature Importanceのグラフからは、\
                        利用料の合計（**Charge**）や予約者の年齢（**Age**）\
                        がキャンセル（**Status**）に影響を与えていることがわかります。",
                },
            }
        )
        I18N_DICT_FOR_TABULAR["correlation_feature_and_target"].update(
            {
                "tutorial": {
                    "en": "**NOTE:** Let's see how feature **Age** affects the **Status**. \
                        It is immediately apparent that the people who cancel are a little older \
                        than the people who stay.",
                    "ja": "**NOTE:** 予約者の年齢（**Age**）がどのようにキャンセル（**Status**）に\
                        影響をあたえているか見てみましょう。キャンセルをする人は滞在する人に比べて年齢がやや高い事がわかります。",
                },
            }
        )

    elif sampledata == "Housing Prices":
        I18N_DICT_FOR_TABULAR["training_data"].update(
            {
                "tutorial": {
                    "en": "**NOTE:** The {source} contains information that influences housing prices, \
                        such as housing specifications and location. This tutorial predicts **'SalePrice** \
                        indicating the sales price of the houses.",
                    "ja": "**NOTE:** {source}は住宅の仕様や立地など住宅価格に影響を与える情報を含んでいます。\
                        このチュートリアルでは、住宅価格を示す **SalePrice** の推論を行います",
                },
            }
        )

        I18N_DICT_FOR_TABULAR["model_details"].update(
            {
                "tutorial": {
                    "en": "**NOTE:** From the Permutation Feature Importance graph of the sample data, \
                        we can see that **OverallQual** has an impact \
                        on price of the houses (**SalePrice**）.",
                    "ja": "**NOTE:** サンプルデータのPermutation Feature Importanceのグラフからは、\
                        住宅の全体的な品質（**OverallQual**）が住宅価格（**SalePrice**）に影響を与えていることがわかります。",
                },
            }
        )
        I18N_DICT_FOR_TABULAR["correlation_feature_and_target"].update(
            {
                "tutorial": {
                    "en": "**NOTE:** Let's see how feature **OverallQual** affects the **SalePrice**. \
                        It is immediately apparent that the higher the value of OverallQual, \
                        the higher the SalePrice.",
                    "ja": "**NOTE:** 物件の全体的な品質（**OverallQual**）がどのように\
                        住宅価格（**SalePrice**）に影響をあたえているか見てみましょう。物件の全体的な品質（**OverallQual**）の値が高いほど\
                        住宅価格（**SalePrice**）が高くなることが一目でわかります。",
                },
            }
        )

    elif sampledata == "Medical Insurance Charges":
        I18N_DICT_FOR_TABULAR["training_data"].update(
            {
                "tutorial": {
                    "en": "**NOTE:** {source} includes information that influences medical expenses, \
                        such as the age, gender, and BMI of the policyholders. \
                            This tutorial predict **charges** indicating medical insurance charges.",
                    "ja": "**NOTE:** {source}は契約者の年齢、性別BMIなど医療保険の請求に影響を与える情報を含んでいます。\
                        このチュートリアルでは、保険の請求額を示す **charges** の推論を行います",
                },
            }
        )

        I18N_DICT_FOR_TABULAR["model_details"].update(
            {
                "tutorial": {
                    "en": "**NOTE:** From the Permutation Feature Importance graph of the sample data, \
                        we can see that **smoker** and **age** have an impact \
                        on **charges**.",
                    "ja": "**NOTE:** サンプルデータのPermutation Feature Importanceのグラフからは、\
                        喫煙の有無（**smoker**）や年齢（**age**）が医療保険の請求額（**charges**）に影響を与えていることがわかります。",
                },
            }
        )
        I18N_DICT_FOR_TABULAR["correlation_feature_and_target"].update(
            {
                "tutorial": {
                    "en": "**NOTE:** Let's see how **smoker** affects the **charges**. \
                        It is immediately apparent that smoker's charges is large.",
                    "ja": "**NOTE:** 喫煙の有無（**smoker**）がどのように医療保険の請求額（charges）に影響をあたえているか見てみましょう。喫煙者の請求額が大きいことが一目でわかります。",
                },
            }
        )

    return I18N_DICT_FOR_TABULAR
