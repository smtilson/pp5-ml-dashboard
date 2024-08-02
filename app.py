from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_eda import page_eda_body
from app_pages.page_hypotheses import page_hypotheses_body
from app_pages.page_hypothesis_1 import page_hypothesis_1_body
from app_pages.page_predictor import page_predictor_body
from app_pages.page_cluster import page_cluster_body
from app_pages.page_adaboost_model import page_adaboost_model_body
from app_pages.page_logistic_model import page_logistic_model_body
from app_pages.page_conclusions import page_conclusions_body

app = MultiPage(app_name="NBA Home Team Wins")

# Add your app pages here using .add_page()
app.add_page("Project Summary", page_summary_body)
app.add_page("Exploratory Data Analysis", page_eda_body)
app.add_page("Predictor", page_predictor_body)
app.add_page("Project Hypotheses and Validation", page_hypotheses_body)
app.add_page("ML: Naive Feature Selection", page_hypothesis_1_body)
app.add_page("ML: Logistic Regression Model", page_logistic_model_body)
app.add_page("ML: Adaptive Boost Model", page_adaboost_model_body)
app.add_page("ML: Cluster Analysis", page_cluster_body)
app.add_page("Project Conclusions", page_conclusions_body)

app.run()  # Run the  app
