"""
This script trains the user-selected ML algorithms and ranks them by
accuracy score, returning the model with the highest accuracy. With the
best-performing model, the user is able to make a prediction with a new,
unknown sample.

To run script: python3 biome.py
(or python3 -W ignore biome.py to ignore warnings)
"""
import pandas as pd
from sklearn import preprocessing

from prep_split_data import data_loader
from prep_split_data import get_one_hot
from prep_split_data import split_train_test
from select_model import evaluate_rank_models
from select_model import get_prediction

print('                                                          ***********'
      '                     ')
print('          \\           |              |                ***************'
      '*******              ')
print('        .%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ********************'
      '*************        ')
print('     -%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% **********************'
      '***************      ')
print('  \\-%%%%%% \033[96m@@@@@@@@@\033[0m %%%% \033[96m@@\033[0m %%%%%%%%%%%'
      '%%%%%%  \033[94m@@@\033[0m ******** \033[94m@@\033[0m *** \033[94m@@@@'
      '@@@@@@\033[0m *********   ')
print('  *%%%%%%% \033[96m@\033[0m %%%%%% \033[96m@@\033[0m %%%%%%%%%%%%%%%%%'
      '%%%% ** \033[94m@@@\033[0m ******* \033[94m@@@\033[0m **'
      '* \033[94m@\033[0m *******************  ')
print(' +%%%%%%%% \033[96m@\033[0m %%%%%% \033[96m@@\033[0m %'
      '% \033[96m@@\033[0m %%%%% \033[96m@@@@\033[0m %%%% *'
      '* \033[94m@@@@\033[0m ***** \033[94m@@@@\033[0m **'
      '* \033[94m@\033[0m *******************  ')
print('_%%%%%%%%% \033[96m@@@@@@@@@@\033[0m %%% \033[96m@@\033[0m %'
      '% \033[96m@@\033[0m %%%% \033[96m@@\033[0m % *'
      '* \033[94m@@ @@\033[0m *** \033[94m@@ @@\033[0m **'
      '* \033[94m@@@@@@@@@@\033[0m************ ')
print(' *%%%%%%%% \033[96m@\033[0m %%%%%%% \033[96m@\033[0m %'
      '% \033[96m@@\033[0m % \033[96m@@\033[0m %%%%%'
      '% \033[96m@\033[0m % ** \033[94m@@\033[0m '
      '* \033[94m@@ @@\033[0m * \033[94m@@\033[0m **'
      '* \033[94m@\033[0m *********************')
print('  #%%%%%%% \033[96m@\033[0m %%%%%% \033[96m@@\033[0m %'
      '% \033[96m@@\033[0m %% \033[96m@@\033[0m %%%% \033[96m@@\033[0m %'
      '% * \033[94m@@\033[0m ** \033[94m@@@\033[0m ** \033[94m@@\033[0m **'
      '* \033[94m@\033[0m *********************')
print('  /+%%%%%% \033[96m@@@@@@@@@@\033[0m %%% \033[96m@@\033[0m %%%'
      '% \033[96m@@@@@@\033[0m  %%% * \033[94m@@\033[0m *** \033[94m@\033[0m '
      '*** \033[94m@@\033[0m *** \033[94m@@@@@@@@@@\033[0m ************')
print('     .%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ***********************'
      '******************** ')
print('        .%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ********************'
      '*******************  ')
print('           /            |              |                 *************'
      '******************   ')
print('                                                                 *****'
      '***************     ')
print('                                                                      '
      '************        ')
print('                                                                      '
      '  ******           ')
print('')

OTU_path = input('Please enter the relative path to the OTU data: ')
print('')
metadata_path = input(
        'Please enter the relative path to the categorical data: ')
print('')
category_labels = list((input(
    'Please list the categorical variables of interest: ')).split(','))
print('')
print('What models would you like to test?')
model_list = list(input(
    'See README.md for abbreviations. Type all if all models should be '
    'tested: ').split(','))
print('')

x_data, y_data = data_loader(OTU_path, metadata_path)
y_output = get_one_hot(category_labels, y_data)
x_train_unscale, x_test_unscale, y_train, y_test = split_train_test(x_data,
                                                                    y_output)
scaler = preprocessing.StandardScaler().fit(x_train_unscale)
x_train = scaler.transform(x_train_unscale)
x_test = scaler.transform(x_test_unscale)

model_out = evaluate_rank_models(x_train, y_train, x_test, y_test, model_list)

model_name = model_out[0]
best_model = model_out[1]

yes_no = input('Would you like to make a prediction with the best model? ')
print('')
while yes_no == 'yes' or yes_no == 'Yes':
    query_path = input(
            'Please enter the path to the data that you would like to make'
            ' a prediction for: ')
    print('')
    query_data_pd = pd.read_table(query_path, index_col=0).T
    query_data = query_data_pd.to_numpy()
    query_data_scale = scaler.transform(query_data)
    get_prediction(query_data_scale, model_name, best_model, category_labels)
    print('')
    yes_no = input('Would you like to make another prediction? ')
    print('')
