import logging
import os

from utils import constants


def project_setup():
    """
    Perfomrs initial setups before the execution of the workflow.
    """
    if not os.path.exists(constants.PLOTS_FOLDER):
        os.mkdir(constants.PLOTS_FOLDER)
        os.mkdir(os.path.join(
            constants.PLOTS_FOLDER,
            "conf_matrix"
        ))
        os.mkdir(os.path.join(
            constants.PLOTS_FOLDER,
            "conf_matrix",
            "svm_tf_idf"
        ))
        os.mkdir(os.path.join(
            constants.PLOTS_FOLDER,
            "conf_matrix",
            "nb"
        ))
        os.mkdir(os.path.join(
            constants.PLOTS_FOLDER,
            "conf_matrix",
            "transformers"
        ))
        os.mkdir(os.path.join(
            constants.PLOTS_FOLDER,
            "nb_hyp_tuning"
        ))

    ## LOGGING SETUP
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler("project_logs/assignment.log"),
            logging.StreamHandler()
        ]
    )