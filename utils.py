
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


def prophet_fit(df, prophet_model, today_index, lookback_days=None, predict_days=21):
    """
    Fit the model to the time-series data and generate forecast for specified time frames

    Args
    ----

    df : pandas DataFrame
        The daily time-series data set contains ds column for
        dates (datetime types such as datetime64[ns]) and y column for numerical values

    prophet_model : Prophet model
        Prophet model with configured parameters

    today_index : int
        The index of the date list in the df dataframe, where Day (today_index-lookback_days)th
        to Day (today_index-1)th is the time frame for training

    lookback_days: int, optional (default=None)
        As described above, use all the available dates until today_index as
        training set if no value assigned

    predict_days: int, optional (default=21)
        Make prediction for Day (today_index)th to Day (today_index+predict_days)th

    Returns
    -------
    fig : matplotlib Figure
        A plot with actual data, predicted values and the interval
    forecast : pandas DataFrame
        The predicted result in a format of dataframe
    prophet_model : Prophet model
        Trained model
    """

    # segment the time frames
    baseline_ts = df['ds'][:today_index]
    baseline_y = df['y'][:today_index]
    if not lookback_days:
        print('Use the data from {} to {} ({} days)'.format(df['ds'][0],
                                                            df['ds'][today_index - 1],
                                                            today_index))
    else:
        baseline_ts = df['ds'][today_index - lookback_days:today_index]
        baseline_y = df.y[today_index - lookback_days:today_index]
        print('Use the data from {} to {} ({} days)'.format(df['ds'][today_index - lookback_days],
                                                            df['ds'][today_index - 1],
                                                            lookback_days))
    print('Predict {} to {} ({} days)'.format(df['ds'][today_index],
                                              df['ds'][today_index + predict_days - 1],
                                              predict_days))

    # fit the model
    prophet_model.fit(pd.DataFrame({'ds': baseline_ts.values,
                                    'y': baseline_y.values}))
    future = prophet_model.make_future_dataframe(periods=predict_days)
    # make prediction
    forecast = prophet_model.predict(future)
    # generate the plot
    fig = prophet_model.plot(forecast)
    return fig, forecast, prophet_model


def prophet_plot(df, fig, today_index, lookback_days=None, predict_days=21, outliers=list()):
    """
    Plot the actual, predictions, and anomalous values

    Args
    ----

    df : pandas DataFrame
        The daily time-series data set contains ds column for
        dates (datetime types such as datetime64[ns]) and y column for numerical values

    fig : matplotlib Figure
        A plot with actual data, predicted values and the interval which we previously obtained
        from Prophet's model.plot(forecast).

    today_index : int
        The index of the date list in the dataframe dividing the baseline and prediction time frames.

    lookback_days : int, optional (default=None)
        Day (today_index-lookback_days)th to Day (today_index-1)th is the baseline time frame for training.

    predict_days : int, optional (default=21)
        Make prediction for Day (today_index)th to Day (today_index+predict_days)th.

    outliers : a list of (datetime, int) tuple
        The outliers we want to highlight on the plot.
    """
    # retrieve the subplot in the generated Prophets matplotlib figure
    ax = fig.get_axes()[0]

    start = 0
    end = today_index + predict_days
    x_pydatetime = df['ds'].dt.to_pydatetime()
    # highlight the actual values of the entire time frame
    ax.plot(x_pydatetime[start:end],
            df.y[start:end],
            color='orange', label='Actual')

    # plot each outlier in red dot and annotate the date
    for outlier in outliers:
        ax.scatter(outlier[0], outlier[1], color='red', label='Anomaly')
        ax.text(outlier[0], outlier[1], str(outlier[0])[:10], color='red')

    # highlight baseline time frame with gray background
    if lookback_days:
        start = today_index - lookback_days
    ax.axvspan(x_pydatetime[start],
               x_pydatetime[today_index],
               color=sns.xkcd_rgb['grey'],
               alpha=0.2)

    # annotate the areas, and position the text at the bottom 5% by using ymin + (ymax - ymin) / 20
    ymin, ymax = ax.get_ylim()[0], ax.get_ylim()[1]
    ax.text(x_pydatetime[int((start + today_index) / 2)], ymin + (ymax - ymin) / 20, 'Baseline area')
    ax.text(x_pydatetime[int((today_index * 2 + predict_days) / 2)], ymin + (ymax - ymin) / 20, 'Prediction area')

    # re-organize the legend
    patch1 = mpatches.Patch(color='red', label='Anomaly')
    patch2 = mpatches.Patch(color='orange', label='Actual')
    patch3 = mpatches.Patch(color='skyblue', label='Predict and interval')
    patch4 = mpatches.Patch(color='grey', label='Baseline area')
    plt.legend(handles=[patch1, patch2, patch3, patch4])
    plt.show()


def get_outliers(df, forecast, today_index, predict_days=21):
    """
    Combine the actual values and forecast in a data frame and identify the outliers

    Args
    ----

    df : pandas DataFrame
        The daily time-series data set contains ds column for
        dates (datetime types such as datetime64[ns]) and y column for numerical values

    forecast : pandas DataFrame
        The predicted result in a dataframe which was previously generated by
        Prophet's model.predict(future)

    today_index : int
        The summary statistics of the right tree node.

    predict_days : int, optional (default=21)
        The time frame we segment as prediction period

    Returns
    -------
    outliers : a list of (datetime, int) tuple
        A list of outliers, the date and the value for each
    df_pred : pandas DataFrame
        The data set contains actual and predictions for the forecast time frame
    """
    df_pred = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(predict_days)
    df_pred.index = df_pred['ds'].dt.to_pydatetime()
    df_pred.columns = ['ds', 'preds', 'lower_y', 'upper_y']
    df_pred['actual'] = df['y'][today_index: today_index + predict_days].values

    # construct a list of outliers
    outlier_index = list()
    outliers = list()
    for i in range(df_pred.shape[0]):
        actual_value = df_pred['actual'][i]
        if actual_value < df_pred['lower_y'][i] or actual_value > df_pred['upper_y'][i]:
            outlier_index += [i]
            outliers.append((df_pred.index[i], actual_value))
            # optional, print out the evaluation for each outlier
            print('=====')
            print('actual value {} fall outside of the prediction interval'.format(actual_value))
            print('interval: {} to {}'.format(df_pred['lower_y'][i], df_pred['upper_y'][i]))
            print('Date: {}'.format(str(df_pred.index[i])[:10]))

    return outliers, df_pred