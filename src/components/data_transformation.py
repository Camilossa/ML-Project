import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    """Configuración para la transformación de datos"""

    # Ruta donde se guardará el objeto preprocessor serializado
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    """Clase principal para manejar la transformación de datos"""

    def __init__(self):
        """Inicializa la configuración de transformación de datos"""
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Crea el objeto de transformación de datos (preprocessor)

        Este método construye un pipeline que:
        - Procesa variables numéricas: imputación + escalado
        - Procesa variables categóricas: imputación + one-hot encoding + escalado

        Returns:
            ColumnTransformer: Objeto que combina ambos pipelines
        """
        try:
            # Definir las columnas numéricas del dataset
            numerical_columns = ["writing_score", "reading_score"]

            # Definir las columnas categóricas del dataset
            categorical_columns = [
                "gender",  # Género del estudiante
                "race_ethnicity",  # Etnia/raza del estudiante
                "parental_level_of_education",  # Nivel educativo de los padres
                "lunch",  # Tipo de almuerzo (gratuito/reducido/estándar)
                "test_preparation_course",  # Si completó curso de preparación
            ]

            # Pipeline para variables numéricas
            num_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    ),  # Rellena valores faltantes con la mediana
                    (
                        "scaler",
                        StandardScaler(),
                    ),  # Normaliza los datos (media=0, std=1)
                ]
            )

            # Pipeline para variables categóricas
            cat_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),  # Rellena con el valor más frecuente
                    (
                        "one_hot_encoder",
                        OneHotEncoder(),
                    ),  # Convierte a variables dummy (0 y 1)
                    (
                        "scaler",
                        StandardScaler(with_mean=False),
                    ),  # Normaliza sin centrar en la media
                ]
            )

            # Registrar información en los logs
            logging.info(f"Columnas categóricas: {categorical_columns}")
            logging.info(f"Columnas numéricas: {numerical_columns}")

            # Combinar ambos pipelines en un solo transformador
            preprocessor = ColumnTransformer(
                [
                    (
                        "num_pipeline",
                        num_pipeline,
                        numerical_columns,
                    ),  # Aplicar pipeline numérico
                    (
                        "cat_pipelines",
                        cat_pipeline,
                        categorical_columns,
                    ),  # Aplicar pipeline categórico
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Ejecuta todo el proceso de transformación de datos

        Este método:
        1. Carga los datos de entrenamiento y prueba
        2. Separa las características (features) de la variable objetivo (target)
        3. Aplica las transformaciones a ambos conjuntos de datos
        4. Guarda el objeto preprocessor para uso futuro
        5. Retorna los datos transformados listos para entrenar el modelo

        Args:
            train_path (str): Ruta del archivo de datos de entrenamiento
            test_path (str): Ruta del archivo de datos de prueba

        Returns:
            tuple: (train_arr, test_arr, preprocessor_path)
                - train_arr: Array de entrenamiento transformado
                - test_arr: Array de prueba transformado
                - preprocessor_path: Ruta del objeto preprocessor guardado
        """
        try:
            # 1. Cargar los datos desde archivos CSV
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Lectura de datos de entrenamiento y prueba completada")

            # 2. Obtener el objeto preprocessor configurado
            logging.info("Obteniendo objeto de preprocesamiento")
            preprocessing_obj = self.get_data_transformer_object()

            # 3. Definir la variable objetivo que queremos predecir
            target_column_name = "math_score"  # Puntuación en matemáticas

            # 4. Separar características (X) de la variable objetivo (y) para entrenamiento
            input_feature_train_df = train_df.drop(
                columns=[target_column_name], axis=1
            )  # Todas las columnas excepto math_score
            target_feature_train_df = train_df[
                target_column_name
            ]  # Solo la columna math_score

            # 5. Separar características (X) de la variable objetivo (y) para prueba
            input_feature_test_df = test_df.drop(
                columns=[target_column_name], axis=1
            )  # Todas las columnas excepto math_score
            target_feature_test_df = test_df[
                target_column_name
            ]  # Solo la columna math_score

            logging.info(
                "Aplicando objeto de preprocesamiento en los dataframes de entrenamiento y prueba"
            )

            # 6. Aplicar transformaciones a los datos de entrenamiento
            # fit_transform: aprende los parámetros (media, std, etc.) y transforma
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )

            # 7. Aplicar transformaciones a los datos de prueba
            # transform: solo transforma usando los parámetros ya aprendidos
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # 8. Combinar características transformadas con la variable objetivo
            # np.c_ concatena horizontalmente los arrays
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]  # Features + target de entrenamiento
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]  # Features + target de prueba

            logging.info("Objeto de preprocesamiento guardado")

            # 9. Guardar el objeto preprocessor para uso futuro (predicciones nuevas)
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            # 10. Retornar los datos transformados y la ruta del preprocessor
            return (
                train_arr,  # Datos de entrenamiento transformados
                test_arr,  # Datos de prueba transformados
                self.data_transformation_config.preprocessor_obj_file_path,  # Ruta del preprocessor guardado
            )

        except Exception as e:
            raise CustomException(e, sys)
