
from optparse import OptionParser
import os
import stats

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import entropy
from collections import Counter
from sklearn.linear_model import LinearRegression as LR

import smoothing
import conversion


# declare frame names
FRAMES = ['Economic',
          'Capacity_and_resources',
          'Morality',
          'Fairness_and_equality',
          'Legality_jurisdiction',
          'Policy_prescription',
          'Crime_and_punishment',
          'Security_and_defense',
          'Health_and_safety',
          'Quality_of_life',
          'Cultural_identity',
          'Public_sentiment',
          'Political',
          'External_regulation',
          'Other']

named_cols = FRAMES + ['Pro', 'Neutral', 'Anti', 'tone', 'stories']


def main():
    usage = "%prog "
    parser = OptionParser(usage=usage)
    parser.add_option('--groupby', dest='groupby', default='month',
                      help='Group by [week|month|quarter|year]: default=%default')
    parser.add_option('--max_polls', dest='max_polls', default=50,
                      help='Maximum number of polls: default=%default')
    parser.add_option('--polls', action="store_true", dest="polls", default=False,
                      help='Predict polls (else predict mood): default=%default')
    parser.add_option('--body', action="store_true", dest="body", default=False,
                      help='Use body codes (else use primary)   : default=%default')
    parser.add_option('--n_periods', dest='n_periods', default=6,
                      help='Number of periods to use for predicting polls: default=%default')
    parser.add_option('--n_surge', dest='n_surge', default=1,
                      help='Number of blocks of time to use for surge comparison: default=%default')
    parser.add_option('--use_intercept', action="store_true", dest="use_intercept", default=False,
                      help="Use an intercept in each model: default=%default")
    parser.add_option('--do_smoothing', action="store_true", dest="do_smoothing", default=False,
                      help="Smooth everything!: default=%default")


    (options, args) = parser.parse_args()

    group_by = options.groupby
    max_polls = int(options.max_polls)
    predict_polls = options.polls
    use_body = options.body
    n_periods = int(options.n_periods)
    n_surge = int(options.n_surge)
    intercept = options.use_intercept
    smooth = options.do_smoothing
    first_year = 1992
    last_year = 2012

    subjects = ['immigration', 'same-sex marriage', 'gun control']
    data_files = ['immigration_with_metadata_2017_05_25.csv', 'ssm_with_metadata_2017_05_25.csv', 'guncontrol_with_metadata_2017_05_25.csv']
    polls_files = ['immigration_polls_dedup_merge.csv', 'samesex_marriage_polls.csv', 'gun_control_for_wcalc_corrected.csv']
    data_dir = os.path.join('..', 'data')

    for s_i, subject in enumerate(subjects):
        print('\n' + subject)
        data_file = os.path.join(data_dir, data_files[s_i])
        polls_file = os.path.join(data_dir, polls_files[s_i])
        run_analysis(subject, data_file, polls_file, first_year, last_year, group_by, predict_polls, use_body, max_polls, n_periods, n_surge, intercept, smooth)


def run_analysis(subject, data_file, polls_file, first_year, last_year, group_by, predict_polls, use_body, max_polls, n_periods, n_surge, intercept, smooth):

    # load framing and tone data
    data = pd.read_csv(data_file, header=0, index_col=0)

    # subselect based on rows (and irrelevant)
    data = data.loc[(data['Year'] >= first_year) & (data['Year'] <= last_year)]

    # create a proper date column
    data['date'] = conversion.ymd_to_dates(data['Year'].ravel(), data['Month'].ravel(), data['Day'].ravel())

    # count each row as one story
    data['stories'] = 1

    # compute tone
    data['tone'] = data['Pro'] - data['Anti']

    # rename p0 or b0 to proper frames
    data = rename_framing_columns(data, use_body)
    data = interact_framing_and_tone(data)

    print("Loaded data for %d articles" % len(data.index))

    # group data
    data['period'] = conversion.dates_to_periods(data['date'], first_year, period=group_by)
    groups = data.groupby('period')

    grouped = pd.DataFrame()
    # sum the stories in each time period
    grouped['stories'] = groups.aggregate(np.sum)['stories']

    # average the tone in each time period (also frames, etc.)
    grouped['tone'] = groups.aggregate(np.mean)['tone']
    for t in ['Pro', 'Anti']:
        grouped[t] = groups.aggregate(np.mean)[t]
    for f in FRAMES:
        grouped[f] = groups.aggregate(np.mean)[f]
        for t in ['Pro', 'Anti']:
            col = f + '_' + t
            grouped[col] = groups.aggregate(np.mean)[col]
    periods = grouped.index

    # convert dates to both a float form and a 0-based index form
    grouped['f_date'] = conversion.periods_to_f_dates(periods, first_year, period=group_by)
    grouped['period'] = grouped.index

    # create a new data frame with no gaps in periods to hold smoothed data
    pred_periods = np.arange(np.min(grouped.period), np.max(grouped.period))
    pred_f_dates = conversion.periods_to_f_dates(pred_periods, first_year, period=group_by)

    df_smoothed = pd.DataFrame()
    df_smoothed['f_date'] = pred_f_dates
    df_smoothed['period'] = pred_periods
    df_smoothed.index = pred_periods
    df_smoothed['stories'] = 0

    # copy the number of stories from grouped to df_smoothed
    for p in df_smoothed.period:
        if p in grouped.index:
            df_smoothed.loc[p, 'stories'] = grouped.loc[p, 'stories']

    # smooth tone, pro, and anti
    print("smoothing tone")
    if smooth:
        tone_smooth = smoothing.local_linear(x=grouped.f_date, y=grouped.tone, pred_range=df_smoothed.f_date, bw='cv_ls')
        df_smoothed['tone'] = tone_smooth
        pro_smooth = smoothing.local_linear(x=grouped.f_date, y=grouped.Pro, pred_range=df_smoothed.f_date, bw='cv_ls')
        df_smoothed['pro'] = pro_smooth
        anti_smooth = smoothing.local_linear(x=grouped.f_date, y=grouped.Anti, pred_range=df_smoothed.f_date, bw='cv_ls')
        df_smoothed['anti'] = anti_smooth

    else:
        df_smoothed['tone'] = grouped.tone
        if df_smoothed['tone'].isnull().any():
            df_smoothed = interpolate_nans(grouped, 'f_date', 'tone', df_smoothed, 'f_date', 'tone')
        df_smoothed['pro'] = grouped.Pro
        if df_smoothed['pro'].isnull().any():
            df_smoothed = interpolate_nans(grouped, 'f_date', 'Pro', df_smoothed, 'f_date', 'pro')
        df_smoothed['anti'] = grouped.Anti
        if df_smoothed['anti'].isnull().any():
            df_smoothed = interpolate_nans(grouped, 'f_date', 'Anti', df_smoothed, 'f_date', 'anti')


    # smooth frames and frame_X_tone interactions
    print("smoothing frames")
    for f in FRAMES:
        print(f)
        if smooth:
            smoothed = smoothing.local_linear(x=grouped.f_date, y=grouped[f], pred_range=df_smoothed.f_date)
            df_smoothed[f] = smoothed
        else:
            df_smoothed[f] = grouped[f]
            if df_smoothed[f].isnull().any():
                df_smoothed = interpolate_nans(grouped, 'f_date', f, df_smoothed, 'f_date', f)

        for t in ['Pro', 'Anti']:
            col = f + '_' + t
            print(col)
            if smooth:
                smoothed = smoothing.local_linear(x=grouped.f_date, y=grouped[col], pred_range=df_smoothed.f_date)
                df_smoothed[col] = smoothed
            else:
                df_smoothed[col] = grouped[col]
                if df_smoothed[col].isnull().any():
                    df_smoothed = interpolate_nans(grouped, 'f_date', col, df_smoothed, 'f_date', col)


    # load polls
    print("loading and smoothing polls")
    polls = pd.read_csv(polls_file)

    # convert date (strings) to Timestamp dates, float dates, and periods
    polls['date'] = [pd.Timestamp(dt.datetime.strptime(str(d), '%m/%d/%y')) for d in polls['Date']]
    polls['f_date'] = conversion.dates_to_f_dates(polls['date'])
    polls['period'] = conversion.dates_to_periods(polls['date'], first_year, period=group_by)

    # exclude very old and very recent polls
    polls = polls[(polls.date > pd.Timestamp(year=1990, month=1, day=1)) & (polls.date < pd.Timestamp.today())]
    # sort by date
    polls = polls.sort_values(by='date')

    print("Loaded %d polls" % len(polls.index))

    # replace N for polls with N=0 with min poll size
    print("%d polls with N=0" % len(polls[polls.N==0]))
    min_nonzero_poll_size = polls.loc[polls.N > 0, 'N'].min()
    print("min non-zero poll size = %d" % min_nonzero_poll_size)
    print("Setting N=0 polls to N=%d" % min_nonzero_poll_size)
    polls.loc[polls.N == 0, 'N'] = min_nonzero_poll_size

    # scale the "Index" variable and rename it value
    polls['value'] = polls['Index'] / 100.0

    # add in indicator variables for each question type (Varnames)
    counter = Counter()
    all_varnames = polls['Varname'].values
    # figure out which are most common
    counter.update(all_varnames)
    varnames_counts = counter.most_common(n=max_polls)
    varnames = []
    for v, c in varnames_counts:
        polls[v] = 0
        polls.loc[polls.Varname == v, v] = 1
        varnames.append(v)

    # exclude all but the most populare max_polls polls
    for v in list(set(all_varnames)):
        if v not in polls.columns:
            polls = polls.loc[polls.Varname != v]

    # use linear regression to question type and time to compute offsets between polls
    X = polls[varnames[1:] + ['f_date']]
    y = polls['value']
    model = LR()
    model.fit(X, y)
    polls['mood'] = polls['value']
    for p_i, poll in enumerate(varnames[1:]):
        # set the "mood" variable to the "value" variable minus question offset
        polls.loc[polls.Varname == poll, 'mood'] = polls.loc[polls.Varname == poll, 'mood'] - model.coef_[p_i]

    # smooth and interpolate mood across the full range of dates
    fit = smoothing.local_linear(x=polls.f_date, y=polls.mood, pred_range=df_smoothed.f_date)
    df_smoothed['mood'] = fit

    print("prepping to run models")
    if predict_polls:
        df = prep_to_predict_polls(polls, df_smoothed, n_periods, n_surge)
    else:
        df = prep_to_predict_mood(df_smoothed, n_surge)

    base_dir = os.path.join('..', 'results')
    #tone_cols = ['tone', 'tone_max_diff', 'tone_dom_diff']
    tone_cols = ['tone']
    #dom_cols = ['dom', 'dom_split', 'below_mean', 'below_mean_split', 'surge', 'surge_split']
    dom_cols = ['dom', 'dom_split', 'below_mean', 'below_mean_split']

    exp_name = ''
    if predict_polls:
        exp_name += 'polls'
    else:
        exp_name += 'mood'
    if use_body:
        exp_name += '_body'
    else:
        exp_name += '_primary'
    exp_name += '_' + group_by
    exp_name += '_' + str(n_periods)
    exp_name += '_' + str(n_surge)
    exp_name += '_polls' + str(max_polls)
    if smooth:
        exp_name += '_smooth'
    if intercept:
        exp_name += '_intercept'

    output_dir = os.path.join(base_dir, exp_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fitted, rmse, model = stats.ols(df=df, target='mood', columns=['prev_mood'], add_intercept=intercept)
    with open(os.path.join(output_dir, 'prev_mood.txt'), 'a') as f:
        f.write('\n\n' + subject + '\n=========\n')
        f.write(model.summary().as_text())
        f.write("\nRMSE=%0.4f" % rmse)


    fitted, rmse, model = stats.ols(df=df, target='mood', columns=['prev_mood', 'surge_diff'], add_intercept=intercept)
    with open(os.path.join(output_dir, 'surge_diff.txt'), 'a') as f:
        f.write('\n\n' + subject + '\n=========\n')
        f.write(model.summary().as_text())
        f.write("\nRMSE=%0.4f" % rmse)

    fitted, rmse, model = stats.ols(df=df, target='mood', columns=['prev_mood', 'surge_diff', 'stories'], add_intercept=intercept)
    with open(os.path.join(output_dir, 'surge_diff__stories.txt'), 'a') as f:
        f.write('\n\n' + subject + '\n=========\n')
        f.write(model.summary().as_text())
        f.write("\nRMSE=%0.4f" % rmse)

    fitted, rmse, model = stats.ols(df=df, target='mood', columns=['prev_mood', 'surge_diff_abs'], add_intercept=intercept)
    with open(os.path.join(output_dir, 'surge_diff_abs.txt'), 'a') as f:
        f.write('\n\n' + subject + '\n=========\n')
        f.write(model.summary().as_text())
        f.write("\nRMSE=%0.4f" % rmse)

    fitted, rmse, model = stats.ols(df=df, target='mood_diff', columns=['surge_diff_abs'], add_intercept=intercept)
    with open(os.path.join(output_dir, 'surge_diff_abs__onMoodDiff.txt'), 'a') as f:
        f.write('\n\n' + subject + '\n=========\n')
        f.write(model.summary().as_text())
        f.write("\nRMSE=%0.4f" % rmse)

    for tone_col in tone_cols:
        df['tone_X_stories'] = df[tone_col] * df['stories']
        df['dom_diff_X_stories'] = (df['dom_pro'].values - df['dom_anti'].values) * df['stories'].values

        fitted, rmse, model = stats.ols(df=df, target='mood', columns=['prev_mood', 'tone', 'surge_diff_abs'], add_intercept=intercept)
        with open(os.path.join(output_dir, 'tone__surge_diff_abs.txt'), 'a') as f:
            f.write('\n\n' + subject + '\n=========\n')
            f.write(model.summary().as_text())
            f.write("\nRMSE=%0.4f" % rmse)

        fitted, rmse, model = stats.ols(df=df, target='mood', columns=['prev_mood', 'tone_X_stories', 'surge_diff_abs'], add_intercept=intercept)
        with open(os.path.join(output_dir, 'tone_X_stories__surge_diff_abs.txt'), 'a') as f:
            f.write('\n\n' + subject + '\n=========\n')
            f.write(model.summary().as_text())
            f.write("\nRMSE=%0.4f" % rmse)

        fitted, rmse, model = stats.ols(df=df, target='mood', columns=['prev_mood', tone_col], add_intercept=intercept)
        with open(os.path.join(output_dir, tone_col + '.txt'), 'a') as f:
            f.write('\n\n' + subject + '\n=========\n')
            f.write(model.summary().as_text())
            f.write("\nRMSE=%0.4f" % rmse)

        fitted, rmse, model = stats.ols(df=df, target='mood', columns=['prev_mood', tone_col, 'stories'], add_intercept=intercept)
        with open(os.path.join(output_dir, tone_col + '__' + 'stories.txt'), 'a') as f:
            f.write('\n\n' + subject + '\n=========\n')
            f.write(model.summary().as_text())
            f.write("\nRMSE=%0.4f" % rmse)

        fitted, rmse, model = stats.ols(df=df, target='mood', columns=['prev_mood', tone_col, 'stories', 'tone_X_stories'], add_intercept=intercept)
        with open(os.path.join(output_dir, tone_col + '__stories__tXs.txt'), 'a') as f:
            f.write('\n\n' + subject + '\n=========\n')
            f.write(model.summary().as_text())
            f.write("\nRMSE=%0.4f" % rmse)

        df['tone_X_stories_scaled'] = df['tone_X_stories'].values / df['tone_X_stories'].max()
        fitted, rmse, model = stats.ols(df=df, target='mood', columns=['prev_mood', 'tone_X_stories_scaled'], add_intercept=intercept)
        with open(os.path.join(output_dir, tone_col + '_X_stories_scaled.txt'), 'a') as f:
            f.write('\n\n' + subject + '\n=========\n')
            f.write(model.summary().as_text())
            f.write("\nRMSE=%0.4f" % rmse)

        df['dom_diff_X_stories_scaled'] = df['dom_diff_X_stories'] / df['dom_diff_X_stories'].max()
        fitted, rmse, model = stats.ols(df=df, target='mood', columns=['prev_mood', 'dom_diff_X_stories_scaled'], add_intercept=intercept)
        with open(os.path.join(output_dir, 'dom_diff_X_stories_scaled.txt'), 'a') as f:
            f.write('\n\n' + subject + '\n=========\n')
            f.write(model.summary().as_text())
            f.write("\nRMSE=%0.4f" % rmse)


        for dom_col in dom_cols:
            df['dom_X_tone'] = df[dom_col] * df[tone_col]
            df['dom_X_tone_X_stories'] = df[dom_col] * df['tone_X_stories']

            fitted, rmse, model = stats.ols(df=df, target='mood', columns=['prev_mood', tone_col, 'stories', dom_col], add_intercept=intercept)
            with open(os.path.join(output_dir, tone_col + '__' + 'stories' + '__' + dom_col + '.txt'), 'a') as f:
                f.write('\n\n' + subject + '\n=========\n')
                f.write(model.summary().as_text())
                f.write("\nRMSE=%0.4f" % rmse)

            fitted, rmse, model = stats.ols(df=df, target='mood', columns=['prev_mood', tone_col, 'stories', 'tone_X_stories', dom_col, 'dom_X_tone'], add_intercept=intercept)
            with open(os.path.join(output_dir, tone_col + '__stories__tXs__' + dom_col + '__dXt.txt'), 'a') as f:
                f.write('\n\n' + subject + '\n=========\n')
                f.write(model.summary().as_text())
                f.write("\nRMSE=%0.4f" % rmse)

            fitted, rmse, model = stats.ols(df=df, target='mood', columns=['prev_mood', 'tone_X_stories', dom_col], add_intercept=intercept)
            with open(os.path.join(output_dir, tone_col + '_X_stories__' + dom_col + '.txt'), 'a') as f:
                f.write('\n\n' + subject + '\n=========\n')
                f.write(model.summary().as_text())
                f.write("\nRMSE=%0.4f" % rmse)

            fitted, rmse, model = stats.ols(df=df, target='mood', columns=['prev_mood', 'tone_X_stories', dom_col, 'dom_X_tone_X_stories'], add_intercept=intercept)
            with open(os.path.join(output_dir, tone_col + '_X_stories__' + dom_col + '__dXtXs.txt'), 'a') as f:
                f.write('\n\n' + subject + '\n=========\n')
                f.write(model.summary().as_text())
                f.write("\nRMSE=%0.4f" % rmse)

            fitted, rmse, model = stats.ols(df=df, target='mood', columns=['prev_mood', 'dom_X_tone_X_stories'], add_intercept=intercept)
            with open(os.path.join(output_dir, tone_col + '_X_stories_X_' + dom_col + '.txt'), 'a') as f:
                f.write('\n\n' + subject + '\n=========\n')
                f.write(model.summary().as_text())
                f.write("\nRMSE=%0.4f" % rmse)


def rename_framing_columns(df, use_body=False):
    columns = list(df.columns)
    for f_i, f in enumerate(FRAMES):
        if use_body:
            col_index = columns.index('b' + str(f_i))
        else:
            col_index = columns.index('p' + str(f_i))
        columns[col_index] = f
    df2 = df.copy()
    df2.columns = columns
    return df2


def interact_framing_and_tone(df):
    df2 = df.copy()
    for f in FRAMES:
        for t in ['Pro', 'Anti']:
            col = f + '_' + t
            df2[col] = df[f] * df[t]
    return df2


def interpolate_nans(input_df, input_x, input_y, output_df, output_x, output_y):
    output_df['interp'] = smoothing.interpolate(x=input_df[input_x], y=input_df[input_y], pred_range=output_df[output_x])
    output_df.loc[output_df[output_y].isnull(), output_y] = output_df.loc[output_df[output_y].isnull(), 'interp']
    return output_df

def wavg(df, col, weight_col):
    return np.sum(df[col] * df[weight_col]) / max(1, df[weight_col].sum())


def wsum(df, col, weight_col):
    return np.sum(df[col] * df[weight_col])


def prep_to_predict_polls(polls, df_smoothed, n_periods, n_surge):
    # see if we can predict deviations from the linear fit based on tone in the previous time window
    #time_window_periods = 1
    #prev_blocks = 1

    # exclude polls after we have tone data
    polls_subset = polls[(polls.period > df_smoothed.period.min() + n_periods * 2) & (polls.period <= df_smoothed.period.max())]
    polls_subset = polls_subset.sort_values(by='date')

    #polls_subset['pro'] = np.NaN
    #polls_subset['anti'] = np.NaN
    #polls_subset['pro_max'] = np.NaN
    #polls_subset['anti_max'] = np.NaN
    #polls_subset['pro_dom'] = np.NaN
    #polls_subset['anti_dom'] = np.NaN
    #polls_subset['prev_mood'] = np.NaN
    #polls_subset['below_mean'] = np.NaN
    #polls_subset['below_mean_split'] = np.NaN

    pro_cols = [f + '_' + 'Pro' for f in FRAMES]
    anti_cols = [f + '_' + 'Anti' for f in FRAMES]
    cols_both = pro_cols + anti_cols

    for i, poll in enumerate(polls_subset.index):
        period = polls_subset.loc[poll, 'period']
        curr_grouped_subset = df_smoothed.loc[(df_smoothed.period <= period) & (df_smoothed.period > period - n_periods)]
        prev_grouped_subset = df_smoothed.loc[(df_smoothed.period <= period - n_periods) & (df_smoothed.period > period - 2 * n_periods * n_surge)]

        # compute the easy ones
        polls_subset.loc[poll, 'tone'] = wavg(curr_grouped_subset, 'tone', 'stories')
        polls_subset.loc[poll, 'pro'] = wavg(curr_grouped_subset, 'pro', 'stories')
        polls_subset.loc[poll, 'anti'] = wavg(curr_grouped_subset, 'anti', 'stories')
        polls_subset.loc[poll, 'stories'] = curr_grouped_subset['stories'].sum()
        polls_subset.loc[poll, 'prev_mood'] = prev_grouped_subset.mood.mean()
        polls_subset.loc[poll, 'mood_diff'] = polls_subset.loc[poll, 'mood'] - polls_subset.loc[poll, 'prev_mood']

        # compute max values for pro and anti
        col_avgs = curr_grouped_subset.mean()
        for c in FRAMES + cols_both:
            col_avgs[c] = wavg(curr_grouped_subset, c, 'stories')

        polls_subset.loc[poll, 'pro_max'] = col_avgs[pro_cols].max()
        polls_subset.loc[poll, 'anti_max'] = col_avgs[anti_cols].max()

        # compute dominance
        avg_values = col_avgs[FRAMES].values
        order = np.argsort(avg_values)
        polls_subset.loc[poll, 'dom'] = avg_values[order[-1]] - avg_values[order[-2]]
        polls_subset.loc[poll, 'below_mean'] = np.sum(avg_values < np.mean(avg_values))

        avg_values = col_avgs[cols_both].values
        order = np.argsort(avg_values)
        polls_subset.loc[poll, 'dom_split'] = avg_values[order[-1]] - avg_values[order[-2]]
        polls_subset.loc[poll, 'below_mean_split'] = np.sum(avg_values < np.mean(avg_values))

        pro_avg_values = col_avgs[pro_cols].values
        order = np.argsort(pro_avg_values)
        pro_dom = pro_avg_values[order[-1]] - pro_avg_values[order[-2]]
        polls_subset.loc[poll, 'dom_pro'] = pro_dom

        anti_avg_values = col_avgs[anti_cols].values
        order = np.argsort(anti_avg_values)
        anti_dom = anti_avg_values[order[-1]] - anti_avg_values[order[-2]]
        polls_subset.loc[poll, 'dom_anti'] = anti_dom

        # compute surge
        col_groups = [FRAMES, pro_cols, anti_cols, cols_both]
        names = ['surge', 'surge_pro', 'surge_anti', 'surge_split']
        for g_i, group in enumerate(col_groups):

            diff_mean = curr_grouped_subset.sum()
            diff_sum = curr_grouped_subset.sum()

            for c in group:
                diff_mean[c] = wavg(curr_grouped_subset, c, 'stories') - wavg(prev_grouped_subset, c, 'stories')
                diff_sum[c] = wsum(curr_grouped_subset, c, 'stories') - wsum(prev_grouped_subset, c, 'stories')

                max_frame = np.argmax(diff_sum[group])
                polls_subset.loc[poll, names[g_i] + '_abs'] = diff_sum[max_frame]
                polls_subset.loc[poll, names[g_i]] = diff_mean[max_frame]

    polls_subset['tone_max_diff'] = polls_subset['pro_max'] - polls_subset['anti_max']
    polls_subset['tone_dom_diff'] = polls_subset['dom_pro'] - polls_subset['dom_anti']
    polls_subset['surge_diff'] = polls_subset['surge_pro'] - polls_subset['surge_anti']
    polls_subset['surge_diff_abs'] = polls_subset['surge_pro_abs'] - polls_subset['surge_anti_abs']

    return polls_subset


def prep_to_predict_mood(df_smoothed, n_surge):
    # take the mood from the previous period
    df_smoothed['prev_mood'] = df_smoothed['mood'].shift().as_matrix()
    df_smoothed['mood_diff'] = df_smoothed['mood'] - df_smoothed['prev_mood']

    # compute tone as a difference of pro and anti
    df_smoothed['tone'] = df_smoothed['pro'] - df_smoothed['anti']

    # compute tone as a differnence of pro_max and anti_max
    pro_cols = [f + '_Pro' for f in FRAMES]
    anti_cols = [f + '_Anti' for f in FRAMES]
    cols_split = pro_cols + anti_cols

    df_smoothed['pro_max'] = df_smoothed[pro_cols].max(axis=1)
    df_smoothed['anti_max'] = df_smoothed[anti_cols].max(axis=1)
    df_smoothed['tone_max_diff'] = df_smoothed['pro_max'] - df_smoothed['anti_max']

    # compute tone as a difference in dominance
    #df_smoothed['dom'] = 0
    #df_smoothed['dom_split'] = 0
    #df_smoothed['dom_pro'] = 0
    #df_smoothed['dom_anti'] = 0
    #df_smoothed['below_mean'] = 0
    #df_smoothed['below_mean_split'] = 0
    #df_smoothed['below_mean_pro'] = 0
    #df_smoothed['below_mean_anti'] = 0

    for i in df_smoothed.index:
        vals = df_smoothed.loc[i, FRAMES].values
        order = np.argsort(vals)
        df_smoothed.loc[i, 'dom'] = vals[order[-1]] - vals[order[-2]]
        df_smoothed.loc[i, 'below_mean'] = np.sum(vals < np.mean(vals))

        vals = df_smoothed.loc[i, cols_split].values
        order = np.argsort(vals)
        df_smoothed.loc[i, 'dom_split'] = vals[order[-1]] - vals[order[-2]]
        df_smoothed.loc[i, 'below_mean_split'] = np.sum(vals < np.mean(vals))

        vals = df_smoothed.loc[i, pro_cols].values
        order = np.argsort(vals)
        df_smoothed.loc[i, 'dom_pro'] = vals[order[-1]] - vals[order[-2]]
        df_smoothed.loc[i, 'below_mean_pro'] = np.sum(vals < np.mean(vals))

        vals = df_smoothed.loc[i, anti_cols].values
        order = np.argsort(vals)
        df_smoothed.loc[i, 'dom_anti'] = vals[order[-1]] - vals[order[-2]]
        df_smoothed.loc[i, 'below_mean_anti'] = np.sum(vals < np.mean(vals))

    df_smoothed['tone_dom_diff'] = df_smoothed['dom_pro'] - df_smoothed['dom_anti']

    # compute tone as a difference in surge
    df_smoothed['surge'] = np.NaN
    df_smoothed['surge_pro'] = np.NaN
    df_smoothed['surge_anti'] = np.NaN
    df_smoothed['surge_split'] = np.NaN
    for enum_i, i in enumerate(df_smoothed.index):
        col_groups = [FRAMES, pro_cols, anti_cols, cols_split]
        names = ['surge', 'surge_pro', 'surge_anti', 'surge_split']
        for g_i, group in enumerate(col_groups):
            if enum_i > n_surge:
                vals_abs = df_smoothed.loc[i, group].values * df_smoothed.loc[i, 'stories']
                vals_percent = df_smoothed.loc[i, group].values
                prev_rows = df_smoothed[(df_smoothed.period < i) & (df_smoothed.period >= i - n_surge)]
                if n_surge == 1:
                    prev_wavg = prev_rows[group]
                    prev_wsum = prev_rows[group].sum()
                    prev_wavg_vals = prev_wavg.values[0, :]
                    for f in group:
                        prev_wsum[f] = wsum(prev_rows, f, 'stories') / float(n_surge)
                    prev_wsum_vals = prev_wsum.values
                else:
                    prev_wavg = prev_rows[group].mean()
                    prev_wsum = prev_rows[group].sum()
                    for f in group:
                        prev_wavg[f] = wavg(prev_rows, f, 'stories') / float(n_surge)
                        prev_wsum[f] = wsum(prev_rows, f, 'stories') / float(n_surge)
                    prev_wsum_vals = prev_wsum.values
                    prev_wavg_vals = prev_wavg.values

                order = np.argsort(vals_abs - prev_wsum_vals)
                df_smoothed.loc[i, names[g_i] + '_abs'] = vals_abs[order[-1]] - prev_wsum_vals[order[-1]]
                df_smoothed.loc[i, names[g_i]] = vals_percent[order[-1]] - prev_wavg_vals[order[-1]]

    df_smoothed['surge_diff'] = df_smoothed['surge_pro'] - df_smoothed['surge_anti']
    df_smoothed['surge_diff_abs'] = df_smoothed['surge_pro_abs'] - df_smoothed['surge_anti_abs']

    return df_smoothed

if __name__ == '__main__':
    main()
