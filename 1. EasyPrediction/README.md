**NOTICE: This is the jupyter notebook converted to a markdown file. The interactive element does not work, thus the fun is gone. I encourage you to download the repository and to try out the notebook yourself to see the results. Cheers!**

![](img/header.png)


# Introduction
**DotA 2** is a a multiplayer online battle arena game developed by Valve.
Two teams, each consisting of 5 players, fighting against each other. Each player picks one out of over 100 heroes (without replacement). The heroes have different abilities and characteristics. Additionally, players can purchase different items. The overall goal is to destroy the base of the opponent. Have a look [at Purge's DotA 2 introduction video](https://www.youtube.com/watch?v=9Szj-CloJiI), if you want to have more information.

After a game is finished, one can download some of the **game statistics** using the [DotA 2 Web API](https://dev.dota2.com/showthread.php?t=47115). The statistics are used to assess player's performance and to find out about enemies. The Web API returns not all available information, however, the data should be enough to run some data science on it.

The goal of this mini-project is to **predict which team is going to win**, given the match details from the Web API. For the sake of simplicity, I am going to use the *scikit-learn* package for machine learning. The decision tree can be easily visualized and is chosen as machine learning model. We can interactively change parameters and see the resulting decision tree immediately. If you want to try out more sophisticated models such as the random forest classifier, you can easily do this, since all classifiers are equally built up in scikit learn. However, you cannot use the visualization then.

Please install the following packages before starting:

- scikit-learn
- pandas
- ipywidgets

Additionally, run the following command to enable ipywidgets for jupyter notebooks:

`jupyter nbextension enable --py widgetsnbextension`

Finally, please install GraphViz and add it to your PATH.

In the following. We are going to step through building the classifier.


# Data Loading

Firstly, the data is collected. A script saves my match history into a directory 'data'. Each match' details are saved into a separate json file. The following codes collect all json-files in the data directory and creates a list `match_history_files`  which contains all (relative) paths to the json files.


```python
import json
import os

DATA_JSON_PATH = 'data'

### Generate data list
match_history_files = []
for item in os.listdir(DATA_JSON_PATH):
    item_full_path = os.path.join(DATA_JSON_PATH, item)
    if os.path.isfile(item_full_path) and item_full_path.lower().endswith('json'):
        match_history_files.append(item_full_path)
print('Found {} json files in data directory'.format(len(match_history_files)))
```

    Found 500 json files in data directory
    

# Feature Extraction

For the data extraction we use one single function which extracts some information of the match history json file and puts it into a dictionary (`json_to_features` function). Each json file has the following structure:

    {
        players[
            {
                account_id,
                player_slot,
                item_0,
                item_1,
                ...
            },
            {
                account_id,
                player_slot,
                item_0,
                item_1,
                ...
            },
            ...
        ],
        
        duration,
        first_blood_time,
        ...
    }

We just use this information and do not further processing. Every dictionary has a float or integer as value, but no lists or other datastructures (the dictionary is "flat"). This is the great advantage over the original json file and simplifies the further process. 

**Idea for further work**: Do some calculations on the data and develop own features. One interesting feature could be the kill/death ratio or gold per minute.


```python
import json

def json_to_features(json_file_path):
    json_obj = json.load(open(json_file_path, 'r'))
    
    features = dict()
    
    
    def _add_feature_if_exists(featurename, prefix, d = json_obj):
        """
        This function takes the value of the json object and stores it in the dictionary under same name.
        However, you can specify a prefix when necessary (i.e. to distinguish players).
        If the feature does not exist in the original json object (thank you valve!), we just don't add it.
        """
        
        if featurename in d:
            features[f'{prefix}{featurename}'] = d[featurename]

    _add_feature_if_exists('duration', 'general_')
    _add_feature_if_exists('first_blood_time', 'general_')
    _add_feature_if_exists('radiant_score', 'general_')
    _add_feature_if_exists('dire_score', 'general_')
    _add_feature_if_exists('radiant_win', 'general_') # This value shall be predicted later
    
    for i, player in enumerate(json_obj['players']):
        
        # This prfix is used to distinguish each players
        player_prefix = f'player_{i}_'

        _add_feature_if_exists('account_id', player_prefix, player)
        _add_feature_if_exists('player_slot', player_prefix, player)
        _add_feature_if_exists('hero_id', player_prefix, player)
        _add_feature_if_exists('kills', player_prefix, player)
        _add_feature_if_exists('deaths', player_prefix, player)
        _add_feature_if_exists('assists', player_prefix, player)
        _add_feature_if_exists('last_hits', player_prefix, player)
        _add_feature_if_exists('denies', player_prefix, player)
        _add_feature_if_exists('gold_per_min', player_prefix, player)
        _add_feature_if_exists('xp_per_min', player_prefix, player)
        _add_feature_if_exists('hero_damage', player_prefix, player)
        _add_feature_if_exists('tower_damage', player_prefix, player)
        _add_feature_if_exists('hero_healing', player_prefix, player)
        _add_feature_if_exists('gold', player_prefix, player)
        _add_feature_if_exists('gold_spent', player_prefix, player)
        for item in range(6):
            _add_feature_if_exists(f'item_{item}', player_prefix, player)
    
    return features
```

Now we **extract the features** for all of our existing matches which we have found previously and which are stored in the `match_history_files` list. We store the data **in a pandas dataframe**, so we could do some data analysis and visualization later easily. You can have a look into the data by using `df.head()`

Furthermore, we make the items categorical by converting them into **one-hot-encodings**. Instead of storing "player one's item has the item id 44" (i.e. store 44 as value), we store "player one's item is not 1", "player one's item is not 2", ..., "player one's item is 44", "player one's item is not 45". We do this, because we want scikit-learn to use the item id as categorical variable. Further information on one-hot-encoding can be found [in this HackerNoon post](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f).

Finally, we **fill missing values** (thanks valve!) with 0s. When looking at the different statistics this is a valid choice. For items 0 means there is no item and the other values quantify the success. If the value is missing, we just assume that the player did not achieve something and we set the value to 0. Notice, that other values such as the mean could be suitable, too.


```python
### Convert json files to panda

import pandas as pd

# First convert all files to dicts
list_of_feature_dicts = [json_to_features(match_history_file) for match_history_file in match_history_files]

# Convert the dictionaries to a pandas dataframe
df = pd.DataFrame(list_of_feature_dicts)

# Just some cleanup --> making items categorical
for i in range(10):
    for j in range(6):
        df[f'player_{i}_item_{j}'] = df[f'player_{i}_item_{j}'].astype('category')
        occ_dummies = pd.get_dummies(df[f'player_{i}_item_{j}'], prefix = f'player_{i}_slot_{j}', drop_first=True)
        df = pd.concat([df.drop(f'player_{i}_item_{j}', axis=1), occ_dummies], axis=1)
        
## Remove nans --> Set them to 0
df = df.fillna(0)
```


```python
## Data Analysis stuff here
## You could if you want to. Have a look, i.e. to df.head()
```

# Classification

Let's do the real classification. Here is some stuff going on which I a going to describe now in more detail:

1. We do a **train/test-split** on our dataset. We use this as a measure to find out if our model is overfitting. Overfitting means, that the model adopted the noise of the dataset and is not able to generalize. The train dataset is used to train the model and the test dataset is used to evaluate the model. During the training process, the test dataset is not used.

2. We define a function **plot_tree**, which takes some parameters which are also used by the decision tree. This is used for the interactive visualization later. The paramters of the plot_tree functions can be set with the interactive visualization. We then train a DecisionTree classifier and calculate the accuracy on the test set.

3. We **train the decision tree**. One point to mention is the **`class_weight='balanced'` parameter**. Because we do not have an equal amount of wins/losts, we need to tell the classifier that the class weights should be balanced. Think about the following case: If I won only 1 out of 100 games, the classifier could simply say "You lose" without doing any real classification. If we do not balance, the classifier would have an accuracy of 99%. However, if the reason for the low amount of wins is only because of a very unfortunate train/test split (image, that in the test set I win 80% of the games), the classifier fails to generalize. In order to overcome this, we tell the classifier to balance the classes. If you want to read more about this, go to the [decision tree documentation page of scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).

3. We calculate the **importance of the different features** of the classifier. This gives insight on how the classifier works internally and which factors are important to win. Notice, that just because you were able to deal a lot of damage to a tower does not automatically imply that you are winning. It is just an indicator, a hint. (It's a bit like with [Correlation does not imply causation](https://www.youtube.com/watch?v=HUti6vGctQM)).


```python
## Classification

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from ipywidgets import interactive

from sklearn.model_selection import train_test_split

### This is part 1 - Splitting up the data into train/test
train, test = train_test_split(df, test_size=0.2)
train_y = train.pop('general_radiant_win')
train_X = train

test_y = test.pop('general_radiant_win')
test_X = test



def plot_tree(crit, split, depth, min_split, min_leaf=0.2):
    
    ## This is part 2. Notice, that here you can put in any other classifier you want :)
    clf = DecisionTreeClassifier(random_state = 0 
          , criterion = crit
          , splitter = split
          , max_depth = depth
          , min_samples_split=min_split
          , min_samples_leaf=min_leaf
          , class_weight='balanced')
    
    clf.fit(train_X, train_y)
    
    # We calculate the score on the testset
    print('Accuracy on test:', clf.score(test_X, test_y))
    
    # Show feature importance
    # We only show the 10 most important factors, ordered by their importance
    feature_name_to_importance = dict()
    for feature_index, feature_importance in enumerate(clf.feature_importances_):
        if feature_importance > 0:
            feature_name_to_importance[train_X.columns.values[feature_index]] = feature_importance

    feature_name_to_importance = sorted(feature_name_to_importance.items(), key=lambda kv: kv[1], reverse=True)[:10]

    print()
    print('Feature importance')
    for key,value in feature_name_to_importance:
        print(key,value)

    
    # Then we create the graph using graphviz
    graph = Source(tree.export_graphviz(clf
          , out_file=None
          , feature_names=train_X.columns.values
          , class_names=['Win', 'Lose'] 
          , filled = True))

    display(SVG(graph.pipe(format='svg')))

    return clf

# This is the interaction widget
inter=interactive(plot_tree 
   , crit = ["gini", "entropy"]
   , split = ["best", "random"]
   , depth=list(range(1,10))#[1,2,3,4]
   , min_split=(0.1,1)
   , min_leaf=(0.1,0.5))

display(inter)
```


<p>Failed to display Jupyter Widget of type <code>interactive</code>.</p>
<p>
  If you're reading this message in Jupyter Notebook or JupyterLab, it may mean
  that the widgets JavaScript is still loading. If this message persists, it
  likely means that the widgets JavaScript library is either not installed or
  not enabled. See the <a href="https://ipywidgets.readthedocs.io/en/stable/user_install.html">Jupyter
  Widgets Documentation</a> for setup instructions.
</p>
<p>
  If you're reading this message in another notebook frontend (for example, a static
  rendering on GitHub or <a href="https://nbviewer.jupyter.org/">NBViewer</a>),
  it may mean that your frontend doesn't currently support widgets.
</p>



# Result and Further Stuff to try out

With this really simple decision tree I was able to get an **accuracy of over 80%** on the test set by setting the features to 9 and setting the min_split and min_leaf values to a minimum. This means, we are able to predict the right winner of a game in 4 out of 5 cases - given the statistics.

If we look at the **feature importance**, we can see that the dire score and the radiant score is of most importance. This is reasonable, because the "score" is the kills which each time has. Usually, the team with the higher kills win. Other features, such as the tower damage is very insightful, too. A lot of tower damage implies that a team was able to get a lot of objectives and most likely achieved to win.

I am sure that you could **increase the accuracy by adding some handcrafted features** such as the kill/death ratio or last hits per minute.

Finally, I think it would be pretty interesting to do a **live prediction for a running game**. This could imply some **time series analysis and neural networks (RNNs for example)**. I might try this out in the future, but first I need to find a reliable way to gathher live data. 

See you :)
