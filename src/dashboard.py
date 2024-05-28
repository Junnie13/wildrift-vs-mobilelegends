# Importing Libraries
from importing import *
warnings.filterwarnings("ignore")

# Streamlit Page Configuration
# Set the page layout and title
st.set_page_config(layout="wide", page_title="Reviews Dashboard")
# Disable a deprecation warning related to pyplot
st.set_option('deprecation.showPyplotGlobalUse', False)
# Display the main title of the dashboard
st.title("Reviews Dashboard")

# Load Datasets
# Load the semi and clean datasets from external sources (CSV files)
semi = pd.read_csv("https://raw.githubusercontent.com/Junnie-FTWB8/files/main/semi_dashboard_play_store_reviews_6_months.csv")
clean = pd.read_csv("https://raw.githubusercontent.com/Junnie-FTWB8/files/main/clean_dashboard_play_store_reviews_6_months.csv")

# Topic Modeling
clean = topic_modeling(clean)

# ---------------- Sidebar for User Input ----------------
# Create a sidebar for user input parameters
with st.sidebar:
    # Display a header for parameter selection
    st.header("Parameters")
    
    # User selects months for data filtering
    month = st.multiselect(
        'Select Month/s (2023)',
        ['All Months', 'May', 'June', 'July', 'August', 'September', 'October', 'November'],
        default=['All Months']
    )

    # Display an error message if no month is selected
    if not month:
        st.error("Please select at least one month.")
    
    # User selects a rating category (All, Low, High)
    bin = st.selectbox(
        'Select Rating Category',
        ['All Ratings','Low Ratings', 'High Ratings']
    )

    # User chooses between Unigram and Bigram grouping
    ngrams = st.radio("Select Grouping", ["Unigram", "Bigram"])
    
    # st.sidebar.subheader("Quick Reference")
    # with st.expander("Opening Details"):
    #     st.write("Content for Opening Category")
    # with st.expander("Usage Details"):
    #     st.write("Content for Usage Category")
    # with st.expander("Updates Details"):
    #     st.write("Content for Updates Category")
    # with st.expander("Management Details"):
    #     st.write("Content for Management Category")
    # with st.expander("Miscellaneous Details"):
    #     st.write("Content for Miscellaneous Category")

# Querying Data Based on User Input
# Filter the dataframes based on user-selected months and rating categories
if 'All Months' in month:
    data_semi = semi.copy()  # For 'All Months', copy the entire dataframe
    data_clean = clean.copy()  # For 'All Months', copy the entire dataframe
else:
    data_semi = semi[semi['month'].isin(month)]  # Filter semi dataframe by selected months
    data_clean = clean[clean['month'].isin(month)]  # Filter clean dataframe by selected months

if bin == 'Low Ratings':
    data_semi = data_semi[data_semi['bin_label'] == 'Low']
    data_clean = data_clean[data_clean['bin_label'] == 'Low']
elif bin == 'High Ratings':
    data_semi = data_semi[data_semi['bin_label'] == 'High']
    data_clean = data_clean[data_clean['bin_label'] == 'High']
elif bin != 'All Ratings':
    data_semi = data_semi[data_semi['bin_label'] == bin]


# ---------------- Main Dashboard Area ----------------

# ROW A (Number of Reviews, Average Rating)
# Calculate the number of reviews and average rating
num_reviews = len(data_clean) - 1
avg_rating = data_clean["rating"].mean()

# Create two columns for displaying metrics
col1, col2 = st.columns(2)

# Card 1: Number of Reviews
with col1:
    st.metric("Number of Reviews", num_reviews)

# Card 2: Average Rating
with col2:
    st.metric("Average Rating", avg_rating)

# ROW B (Rating Distribution and Daily Distribution)
# Create two columns for displaying charts
col1, col2 = st.columns(2)

# Card 1: Rating Distribution
with col1:
    st.header("Rating Distribution")
    rating_counts = data_clean['rating'].value_counts().sort_index()
    st.bar_chart(rating_counts)

# Card 2: Daily Rating Distribution
with col2:
    st.header(f'Daily Rating Distribution - {month} ({bin})')
    daily_rating_counts = data_clean.groupby(['date', 'rating']).size().unstack(fill_value=0)
    st.area_chart(daily_rating_counts)

# ROW C (Topic Frequency - with Unigram and Bigram tabs)
# Create two columns for displaying content
col1, col2, = st.columns([2,1])
# Calculate cluster counts
cluster_counts = data_clean['cluster'].value_counts().reset_index()
cluster_counts.columns = ['Cluster', 'Count']

# Create lists to store results
clusters = []
top_words = []
top_phrases = []
cluster_titles = []  # New list to store cluster titles

for cluster in cluster_counts['Cluster']:
    top_unigrams = unigram_frequency_cluster_top(data_clean, cluster, top=1)
    top_bigrams = bigram_frequency_cluster_top(data_clean, cluster, top=1)

    # Collect results in lists
    clusters.append(cluster)
    top_words.append(top_unigrams.iloc[0]['Word'])
    top_phrases.append(top_bigrams.iloc[0]['Bigram'])

    # Assign titles based on top words
    if top_unigrams.iloc[0]['Word'] == 'open':
        cluster_title = 'App Launching'
    elif top_unigrams.iloc[0]['Word'] == 'update':
        cluster_title = 'App Updates'
    elif top_unigrams.iloc[0]['Word'] == 'account':
        cluster_title = 'Acc Management'
    elif top_unigrams.iloc[0]['Word'] == 'use':
        cluster_title = 'App Usability'
    else:
        cluster_title = 'Miscellaneous'

    cluster_titles.append(cluster_title)
cluster_counts['cluster_title'] = cluster_titles

# Cluster Distribution
with col1:
    st.subheader('Themes/Topics')

    # Visualize Cluster Distribution
    # Create a horizontal bar chart using Altair with 'cluster_title' as the y-axis
    chart = alt.Chart(cluster_counts).mark_bar().encode(
        x='Count:Q',
        y=alt.Y('cluster_title:N', sort='-x')
    )
    # Display the chart in Streamlit
    st.altair_chart(chart, use_container_width=True)

with col2:
    # # Display the 'clean' dataframe
    # st.subheader('Filtered Dataframe')
    # st.dataframe(data_clean)
    
    # Create DataFrame without the index column
    result_df = pd.DataFrame({
        'Cluster Title': cluster_titles,
        'Top Word': top_words,
        'Top Phrase': top_phrases
    })
    # Display the table
    st.dataframe(result_df, hide_index=True)

# ROW D (Word Cloud - with Unigram and Bigram tabs)
# Display a subheader for the Word Cloud section
st.subheader("Word Cloud")

# Content based on selected tab (Unigram or Bigram)
if ngrams == "Unigram":
    combined_text = ' '.join(data_clean['review_clean'])
    wordcloud = WordCloud(background_color='white').generate(combined_text)
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
elif ngrams == "Bigram":
    # Initialize an empty list to store all bigrams
    all_bigrams = []
    # Iterate over each review in the list
    for review in data_clean['review_clean']:
        # Tokenize the review into words
        words = nltk.tokenize.wordpunct_tokenize(review)
        # Generate bigrams from the list of words
        bi_grams = list(bigrams(words))
        # Convert bigrams to strings and join the words with a space
        bi_grams_str = [' '.join(gram) for gram in bi_grams]
        # Extend the list of all bigrams with the bigrams from the current review
        all_bigrams.extend(bi_grams_str)
    # Count occurrences of each bigram
    counter_bigrams = Counter(all_bigrams)
    most_frequent_bigrams = counter_bigrams.most_common(30)
    # Create a string combining bigrams based on their count
    bigram_review_words = ' '.join([' '.join([gram] * count) for gram, count in most_frequent_bigrams])
    # Plot the WordCloud image
    wordcloud = WordCloud(background_color='white', min_font_size=10).generate(bigram_review_words)
    plt.figure(figsize=(7, 7), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(plt)

# ROW E (Coefficients)
# Mapping 1 to low rating and 2, 3 to high rating
data_clean['rating_category'] = data_clean['bin_rating'].map({1: 'Low', 2: 'Low', 3: 'High'})

# Mapping Low rating as 0 and High rating as 1
data_clean['binary_rating'] = data_clean['rating_category'].map({'Low': 0, 'High': 1})

# Drop the intermediate column 'rating_category' if you don't need it
df2 = data_clean.drop('rating_category', axis=1)

X = df2.review_clean
y = df2.binary_rating
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

warnings.filterwarnings("ignore", category=FitFailedWarning)

# Creating a pipeline with TfidfVectorizer and Logistic Regression
pipeline = Pipeline([
    ('vect', TfidfVectorizer(ngram_range=(2, 2))),
    ('clf', LogisticRegression(max_iter=1000))  # Increase max_iter if necessary
])

# Define the parameters you want to search through for unigrams and bigrams
parameters = {
    'clf__C': [1.0, 10.0, 100.0],  # Regularization parameter
    'clf__solver': ['lbfgs', 'liblinear'],  # Solvers for Logistic Regression
    'vect__use_idf': [True]
}

# Create GridSearchCV to search for best parameters
grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, error_score='raise')

try:
    # Fit the model
    grid_search.fit(X_train, y_train)

    # Predict with the best model
    y_pred = grid_search.predict(X_test)

except Exception as e:
    print(f"An error occurred during grid search:\n{str(e)}")

# Get feature names after fitting the vectorizer
feature_names = grid_search.best_estimator_.named_steps['vect'].get_feature_names_out()

# Assuming 'clf' in your pipeline represents the Logistic Regression model
logistic_regression_coef = grid_search.best_estimator_.named_steps['clf'].coef_

# Bigrams coefficients
# bigrams_coef = logistic_regression_coef[0][len(feature_names):]
bigrams_coef = logistic_regression_coef[0]    # No need to use Len since the set tfidvectorizer is for Bigrams only. It isn't considering both unigrams and bigrams like in the previous rating prediction

# Display coefficients with corresponding features for bigrams
bigrams_coef_with_features = list(zip(feature_names, bigrams_coef))
sorted_bigrams_coef_with_features = sorted(bigrams_coef_with_features, key=lambda x: x[1], reverse=True)

top_n = 3  # Set the number of top coefficients to display

top_features = []
top_coeffs = []

bottom_features = []
bottom_coeffs = []

for feature, coef in sorted_bigrams_coef_with_features[:top_n]:
    top_features.append(feature)
    top_coeffs.append(coef)

for feature, coef in sorted_bigrams_coef_with_features[-top_n:]:
    bottom_features.append(feature)
    bottom_coeffs.append(coef)

col1, col2 = st.columns(2)

# for unigrams, not functional yet
if ngrams == "Unigram":
    with col1:
        st.subheader("Top Features")
        # Create a bar chart for top coefficients
        st.bar_chart(pd.DataFrame({'Top Features': top_features, 'Top Coefficients': top_coeffs}).set_index('Top Features'), color = "#B2FF66")

    with col2:
        st.subheader("Bottom Features")
        # Create a bar chart for bottom coefficients
        st.bar_chart(pd.DataFrame({'Bottom Features': bottom_features, 'Bottom Coefficients': bottom_coeffs}).set_index('Bottom Features'), color = "#FF6666")

if ngrams == "Bigram":
    with col1:
        st.subheader("Top Features")
        # Create a bar chart for top coefficients
        st.bar_chart(pd.DataFrame({'Top Features': top_features, 'Top Coefficients': top_coeffs}).set_index('Top Features'), color = "#B2FF66")

    with col2:
        st.subheader("Bottom Features")
        # Create a bar chart for bottom coefficients
        st.bar_chart(pd.DataFrame({'Bottom Features': bottom_features, 'Bottom Coefficients': bottom_coeffs}).set_index('Bottom Features'), color = "#FF6666")
