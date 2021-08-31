# The following codes are adapted from ab_analysis.py.
import sys
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def main():
    searchdata_file = sys.argv[1]
    data = pd.read_csv(searchdata_file)
    # Anova and F test.
    # The following codes are adapted from https://towardsdatascience.com/anova-tukey-test-in-python-b3082b6e6bda .
    print("-----------------------ANOVA and F Test-----------------------------------------")
    fvalue, pvalue = stats.f_oneway(data['qs1'],
                                data['qs2'], 
                                data['qs3'],
                                data['qs4'],
                                data['qs5'],
                                data['merge1'],
                                data['partition_sort'])
    print(f"Results of ANOVA test:\n The F-statistic is: {fvalue}\n The p-value is: {pvalue}")
    if(pvalue < 0.05):
        print("Given the p-value is less than significance level, we have strong evidence against the null hypothesis of ANOVA test. Hence,there is difference in means.")
    else:
        print("Given the p-value is not less than significance level, we don't have strong evidence against the null hypothesis of ANOVA test. Hence,there is no difference in means.")
    print("--------------------------------------------------------------------------------")
    print("-----------------------Tukey's HSD Test-----------------------------------------")
    # Tukey's HSD test.
    # The following code is adapted from https://pandas.pydata.org/docs/reference/api/pandas.melt.html .
    horizontal_data = pd.melt(data)
    # The following codes are adapted from https://towardsdatascience.com/anova-tukey-test-in-python-b3082b6e6bda .
    
    m_comp = pairwise_tukeyhsd(endog = horizontal_data['value'], groups = horizontal_data['variable'], alpha = 0.05)
    print(m_comp)
    print("--------------------------------------------------------------------------------")
    print("------------------------------Ranking-------------------------------------------")
    mean_running = data.mean()
    row = ['qs1', 'qs2', 'qs3', 'qs4', 'qs5', 'merge1', 'partition_sort']
    col = ['Avaerage Running Time']
    mean_table = pd.DataFrame(mean_running, index = row, columns = col)
    mean_table['Rank'] = mean_table['Avaerage Running Time'].rank(ascending = True).astype('int')
    mean_table = mean_table.sort_values(['Avaerage Running Time'],ascending = True)
    print(mean_table)
    print("--------------------------------------------------------------------------------")

if __name__ == '__main__':
    main()
