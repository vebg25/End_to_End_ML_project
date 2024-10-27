import sys
import os 
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from src.utils import *

from src.exception import CustomException
from src.logger import logging

class DataTransformationConfig:
  preprocessor_obj_file_path=os.path.join('artifact','preprocessing.pkl')

class DataTransformation:
  def __init__(self):
    self.data_transformation_config=DataTransformationConfig()

  def get_data_transformer_object(self):
    """This is for data transformation"""
    try:
      numerical_columns=['writing_score','reading_score']
      categorical_columns=['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

      num_pipeline=Pipeline(
        steps=[
          ("imputer",SimpleImputer(strategy='median')),
          ('scaler',StandardScaler(with_mean=False))
        ]
      )

      cat_pipeline=Pipeline(
        steps=[
          ('imputer',SimpleImputer(strategy='most_frequent')),
          ("one_hot_encoder",OneHotEncoder()),
          ('scaler',StandardScaler(with_mean=False))
        ]
      )
      logging.info('Categorical and numerical columns done')

      preprocessor=ColumnTransformer(
        [
          ('num_pipeline',num_pipeline,numerical_columns),
          ('cat_pipeline',cat_pipeline,categorical_columns)
        ]
      )

      return preprocessor
    except Exception as e:
      raise CustomException(e,sys)
    
  def initiate_data_transformation(self,train_path,test_path):
    try:
      train_df=pd.read_csv(train_path)
      test_df=pd.read_csv(test_path)
      logging.info("read test and train data")

      preprocessing_obj=self.get_data_transformer_object()
      target_column='math_score'
      numerical_columns=['writing_score','reading_score']
      
      input_feature_df_train=train_df.drop(columns=[target_column],axis=1)
      target_feature_train_df=train_df[target_column]
      input_feature_df_test=test_df.drop(columns=[target_column],axis=1)
      target_feature_test_df=test_df[target_column]

      logging.info(f"Applying preprocessing object on train and test")

      input_feature_train_array=preprocessing_obj.fit_transform(input_feature_df_train)
      input_feature_test_array=preprocessing_obj.transform(input_feature_df_test)

      train_arr=np.c_[input_feature_train_array, np.array(target_feature_train_df)]
      test_arr=np.c_[input_feature_test_array, np.array(target_feature_test_df)]

      logging.info(f"Saved preprocessing object")

      save_object(
        file_path=self.data_transformation_config.preprocessor_obj_file_path,
        obj=preprocessing_obj
      )

      return (
        train_arr,
        test_arr,
        self.data_transformation_config.preprocessor_obj_file_path
      )


    except Exception as e:
      raise CustomException(e,sys)
      