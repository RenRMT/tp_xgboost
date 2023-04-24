# install.packages(c("xgboost", "DiagrammeR")
library(xgboost)
library(DiagrammeR)

# load data ---------------------------------------------------------------

data(agaricus.train, package = "xgboost")
data(agaricus.test, package = "xgboost")


# training ----------------------------------------------------------------
# outputs a sparse matrix
basic_train_sparse <- xgboost(
  data = agaricus.train$data,
  label = agaricus.train$label,
  max.depth = 2,
  eta = 1,
  nthread = 2,
  nrounds = 4,
  objective = "binary:logistic"
)

# outputs a dense matrix (results are the same)
basic_train_dense <- xgboost(
  data = as.matrix(agaricus.train$data),
  label = agaricus.train$label,
  max.depth = 2,
  eta = 1,
  nthread = 2,
  nrounds = 4,
  objective = "binary:logistic"
)

# xgb.Dmatrix combines training data and labels
dtrain <- xgb.DMatrix(data = agaricus.train$data, label = agaricus.train$label)

basic_train_dmatrix <- xgboost(
  data = dtrain,
  max.depth = 2,
  eta = 1,
  nthread = 2,
  nrounds = 4,
  objective = "binary:logistic"
)

# verbose output to help with parameter tuning
basic_train_verbose <- xgboost(
  data = dtrain,
  max.depth = 2,
  eta = 1,
  nthread = 2,
  nrounds = 4,
  objective = "binary:logistic",
  verbose = 2
)

# prediction --------------------------------------------------------------
# predictions on test dataset with predict(), note that output is not (yet)
# a binary classification
pred <- predict(basic_train_dmatrix, agaricus.test$data)

pred_binary <- as.numeric(pred > 0.5)

# classes are balanced, so test error can be meaningful for interpreting model
# performance
table(pred_binary, agaricus.test$label)

test_error <- mean(pred_binary != agaricus.test$label)

# checking for overfitting ------------------------------------------------
dtrain <- xgb.DMatrix(data = agaricus.train$data, label = agaricus.train$label)
dtest <- xgb.DMatrix(data = agaricus.test$data, label = agaricus.test$label)

watchlist <- list(train = dtrain, test = dtest)

bst <- xgb.train(
  data = dtrain,
  max.depth = 2,
  eta = 1,
  nthread = 2,
  nrounds = 4,
  watchlist = watchlist,
  objective = "binary:logistic",
  eval.metric = "error",
  eval.metric = "logloss"
)


# attempting a linear boost model (it seems to do a bit better)
basic_train_linear <- xgb.train(
  data=dtrain, 
  booster = "gblinear", 
  nthread = 2, 
  nrounds=4, 
  watchlist=watchlist, 
  eval.metric = "error", 
  eval.metric = "logloss", 
  objective = "binary:logistic"
  )

# feature importance ------------------------------------------------------
# we learn that smelliness is the most important predictor for whether a 
# mushroom is edible or not (science!)
importance_matrix <- xgb.importance(model = bst)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix)


xgb.dump(bst, with_stats = TRUE)
xgb.plot.tree(model = bst)
