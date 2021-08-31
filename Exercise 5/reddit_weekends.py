import sys
import pandas as pd
import numpy as np
import datetime
from scipy import stats
import matplotlib.pyplot as plt


OUTPUT_TEMPLATE = (
    "Initial T-test p-value: {initial_ttest_p:.3g}\n"
    "Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"
    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
    "Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"
    "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
    "Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"
    "Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"
    "Weekly T-test p-value: {weekly_ttest_p:.3g}\n"
    "Mann-Whitney U-test p-value: {utest_p:.3g}"
)


def main():
    reddit_counts = sys.argv[1]
    counts = pd.read_json(sys.argv[1], lines=True)
    counts = counts[(counts['date'] >= '2012-01-01') & (counts['date'] <= '2013-12-31')] # Values in 2012 and 2013.
    counts = counts[(counts['subreddit'] == 'canada')] # Values in Canada.

    weekday = counts[(counts['date'].map(lambda x: datetime.date.weekday(x) == 0)) | (counts['date'].map(lambda x: datetime.date.weekday(x) == 1)) | (counts['date'].map(lambda x: datetime.date.weekday(x) == 2)) |
                     (counts['date'].map(lambda x: datetime.date.weekday(x) == 3)) | (counts['date'].map(lambda x: datetime.date.weekday(x) == 4))]
    weekend = counts[(counts['date'].map(lambda x: datetime.date.weekday(x) == 5)) | (counts['date'].map(lambda x: datetime.date.weekday(x) == 6))]

    initial_weekday = weekday['comment_count']
    initial_weekend = weekend['comment_count']

    # print('Weekday comments posted is %d.' %)

    # Student's T-Test.
    initial_ttest_p = stats.ttest_ind(initial_weekday, initial_weekend).pvalue
    initial_weekday_normality_p = stats.normaltest(initial_weekday).pvalue
    initial_weekend_normality_p = stats.normaltest(initial_weekend).pvalue
    initial_levene_p = stats.levene(initial_weekday, initial_weekend).pvalue
    # weekday['comment_count'].hist()
    # weekend['comment_count'].hist()
    # plt.title('Fix 1.0')
    # plt.show()


    # Fix 1: transforming data might save us.
    '''
    # Fix 1.1: log.
    log_weekday = initial_weekday.apply(np.log)
    log_weekend = initial_weekend.apply(np.log)
    weekday['comment_count'].apply(np.log).hist()
    weekend['comment_count'].apply(np.log).hist()
    plt.title('Fix 1.1: log.')
    # The above codes are adapted from https://datascience.stackexchange.com/questions/22957/am-i-doing-a-log-transformation-of-data-correctly.
    log_weekday_normality_p = stats.normaltest(log_weekday).pvalue
    log_weekend_normality_p = stats.normaltest(log_weekend).pvalue
    log_levene_p = stats.levene(log_weekday, log_weekend).pvalue
    print('Summary of Fix 1.1: log.')
    print('Weekday normality pvalue is %f.' % log_weekday_normality_p)
    print('Weekend normality pvalue is %f.' % log_weekend_normality_p)
    print('Leneve pvalue is %f.' % log_levene_p)
    print()
    '''

    '''
    # Fix 1.2: exp.
    exp_weekday = initial_weekday.apply(np.exp)
    exp_weekend = initial_weekend.apply(np.exp)
    # weekday['comment_count'].apply(np.exp).hist()
    # weekend['comment_count'].apply(np.exp).hist()
    plt.title('Fix 1.2: exp.')
    plt.show()
    # The above codes are adapted from https://datascience.stackexchange.com/questions/22957/am-i-doing-a-log-transformation-of-data-correctly.
    exp_weekday_normality_p = stats.normaltest(exp_weekday).pvalue
    exp_weekend_normality_p = stats.normaltest(exp_weekend).pvalue
    exp_levene_p = stats.levene(exp_weekday, exp_weekend).pvalue
    print('Summary of Fix 1.2: exp.')
    print('Weekday normality pvalue is %f.' % exp_weekday_normality_p)
    print('Weekend normality pvalue is %f.' % exp_weekend_normality_p)
    print('Leneve pvalue is %f.' % exp_levene_p)
    print()
    '''

    # Fix 1.3: sqrt.
    sqrt_weekday = initial_weekday.apply(np.sqrt)
    sqrt_weekend = initial_weekend.apply(np.sqrt)
    # weekday['comment_count'].apply(np.sqrt).hist()
    # weekend['comment_count'].apply(np.sqrt).hist()
    # plt.title('Fix 1.3: sqrt.')
    # plt.show()
    # The above codes are adapted from https://datascience.stackexchange.com/questions/22957/am-i-doing-a-log-transformation-of-data-correctly.
    sqrt_weekday_normality_p = stats.normaltest(sqrt_weekday).pvalue
    sqrt_weekend_normality_p = stats.normaltest(sqrt_weekend).pvalue
    sqrt_levene_p = stats.levene(sqrt_weekday, sqrt_weekend).pvalue
    # print('Summary of Fix 1.3: sqrt.')
    # print('Weekday normality pvalue is %f.' % sqrt_weekday_normality_p)
    # print('Weekend normality pvalue is %f.' % sqrt_weekend_normality_p)
    # print('Leneve pvalue is %f.' % sqrt_levene_p)
    # print()

    '''
    # Fix 1.4: squared.
    squared_weekday = initial_weekday**2
    squared_weekend = initial_weekend**2
    squared_weekday.hist()
    squared_weekend.hist()
    plt.title('Fix 1.4: squared.')
    plt.show()
    # The above codes are adapted from https://datascience.stackexchange.com/questions/22957/am-i-doing-a-log-transformation-of-data-correctly.
    squared_weekday_normality_p = stats.normaltest(squared_weekday).pvalue
    squared_weekend_normality_p = stats.normaltest(squared_weekend).pvalue
    squared_levene_p = stats.levene(squared_weekday, squared_weekend).pvalue
    print('Summary of Fix 1.4: squared.')
    print('Weekday normality pvalue is %f.' % squared_weekday_normality_p)
    print('Weekend normality pvalue is %f.' % squared_weekend_normality_p)
    print('Leneve pvalue is %f.' % squared_levene_p)
    print()
    '''

    # Fix 2: the Central Limit Theorem might save us.
    # Combine all weekdays and weekend days from each year/week pair and take the mean of their (non-transformed) counts.
    format_weekday = weekday['date'].apply(datetime.date.isocalendar)
    format_weekday = format_weekday.apply(pd.Series)
    format_weekend = weekend['date'].apply(datetime.date.isocalendar)
    format_weekend = format_weekend.apply(pd.Series)
    format_weekday = format_weekday[[0,1]]
    format_weekend = format_weekend[[0,1]]
    format_date = ['year', 'week']
    format_weekday.columns = format_date
    format_weekend.columns = format_date
    weekday = pd.concat([weekday, format_weekday], axis = 'columns', ignore_index = False)
    weekend = pd.concat([weekend, format_weekend], axis = 'columns', ignore_index = False)
    mean_weekday = weekday.groupby(format_date).mean()['comment_count']
    mean_weekend = weekend.groupby(format_date).mean()['comment_count']
    # Check these values for normality and equal variance. 
    weekly_weekday_normality_p = stats.normaltest(mean_weekday).pvalue
    weekly_weekend_normality_p = stats.normaltest(mean_weekend).pvalue
    weekly_levene_p = stats.levene(mean_weekday, mean_weekend).pvalue
    weekly_ttest_p = stats.ttest_ind(mean_weekday, mean_weekend).pvalue

    # Fix 3: a non-parametric test might save us.
    utest_p = stats.mannwhitneyu(initial_weekday, initial_weekend, alternative = 'two-sided').pvalue

    print(OUTPUT_TEMPLATE.format(
        initial_ttest_p = initial_ttest_p,
        initial_weekday_normality_p = initial_weekday_normality_p,
        initial_weekend_normality_p = initial_weekend_normality_p,
        initial_levene_p = initial_levene_p,
        transformed_weekday_normality_p = sqrt_weekday_normality_p,
        transformed_weekend_normality_p = sqrt_weekend_normality_p,
        transformed_levene_p = sqrt_levene_p,
        weekly_weekday_normality_p = weekly_weekday_normality_p,
        weekly_weekend_normality_p = weekly_weekend_normality_p,
        weekly_levene_p = weekly_levene_p,
        weekly_ttest_p = weekly_ttest_p,
        utest_p = utest_p,
    ))


if __name__ == '__main__':
    main()
