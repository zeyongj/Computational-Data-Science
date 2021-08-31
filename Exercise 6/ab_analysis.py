import sys
import pandas as pd
import numpy as np
from scipy import stats

OUTPUT_TEMPLATE = (
    '"Did more/less users use the search feature?" p-value:  {more_users_p:.3g}\n'
    '"Did users search more/less?" p-value:  {more_searches_p:.3g} \n'
    '"Did more/less instructors use the search feature?" p-value:  {more_instr_p:.3g}\n'
    '"Did instructors search more/less?" p-value:  {more_instr_searches_p:.3g}'
)


def main():
    searchdata_file = sys.argv[1]
    data = pd.read_json(searchdata_file, orient = 'records', lines = True) # Adapted from the instruction.

    # For more users.
    even = (data['uid']%2 == 0)
    odd = (data['uid']%2 == 1)
    even_numbered_uid = data[even]
    odd_numbered_uid = data[odd]

    # The following codes are adapted from https://stackabuse.com/python-get-number-of-elements-in-a-list .
    all_even_uid_once = even_numbered_uid[even_numbered_uid['search_count'] >= 1]
    all_even_uid_never = even_numbered_uid[even_numbered_uid['search_count'] == 0]
    all_odd_uid_once = odd_numbered_uid[odd_numbered_uid['search_count'] >= 1]
    all_odd_uid_never = odd_numbered_uid[odd_numbered_uid['search_count'] == 0]
    ## Fining numbers to preceed.
    even_uid_once = len(all_even_uid_once)
    even_uid_never = len(all_even_uid_never)
    odd_uid_once = len(all_odd_uid_once)
    odd_uid_never = len(all_odd_uid_never)
    
    # The following codes are adapted from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html .
    obs1 = [[even_uid_once, even_uid_never], [odd_uid_once, odd_uid_never]]
    chi1, more_users_p, dof1, ex1 = stats.chi2_contingency(obs1, correction = True)

    even_uid_stat = even_numbered_uid['search_count']
    odd_uid_stat = odd_numbered_uid['search_count']
    
    more_searches_p = stats.mannwhitneyu(odd_uid_stat,even_uid_stat, alternative = 'two-sided').pvalue
       
    # For more instructors.
    even_instr = (even_numbered_uid["is_instructor"] == True)
    odd_instr = (odd_numbered_uid["is_instructor"] == True)
    even_numbered_instructor = even_numbered_uid[even_instr]
    odd_numbered_instructor = odd_numbered_uid[odd_instr]
    
    # The following codes are adapted from https://stackabuse.com/python-get-number-of-elements-in-a-list .
    all_even_instructor_once = even_numbered_instructor[even_numbered_instructor['search_count'] >= 1]
    all_even_instructor_never = even_numbered_instructor[even_numbered_instructor['search_count'] == 0]
    all_odd_instructor_once = odd_numbered_instructor[odd_numbered_instructor['search_count'] >= 1]
    all_odd_instructor_never = odd_numbered_instructor[odd_numbered_instructor['search_count'] == 0]
    ## Fining numbers to preceed.
    even_instructor_once = len(all_even_instructor_once)
    even_instructor_never = len(all_even_instructor_never)
    odd_instructor_once = len(all_odd_instructor_once)
    odd_instructor_never = len(all_odd_instructor_never)
    
    # The following codes are adapted from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html .
    obs2 = [[even_instructor_once, even_instructor_never], [odd_instructor_once, odd_instructor_never]]
    chi2, more_instr_p, dof2, ex2 = stats.chi2_contingency(obs2, correction = True)

    even_instr_stat = even_numbered_instructor['search_count']
    odd_instr_stat = odd_numbered_instructor['search_count']
    
    more_instr_searches_p = stats.mannwhitneyu(odd_instr_stat, even_instr_stat, alternative = 'two-sided').pvalue    

    print(OUTPUT_TEMPLATE.format(
        more_users_p = more_users_p,
        more_searches_p = more_searches_p,
        more_instr_p = more_instr_p,
        more_instr_searches_p = more_instr_searches_p,
    ))


if __name__ == '__main__':
    main()
