c_basic_info = ['dx', 'age', 'gender', 'racenew_4grp', 'educ', 'educ2', 'marstat2', 'bmi',
                'smoke', 'Income2', 'work4grp2', 'insurance', 'Any_Comp']

c_self_eval = ['treat_pref_ba', 'better_worse_ba', 'episode4',
               'Episode_fixd', 'WorkLift', 'WorkMiss', 'GuessSSU', 'GuessSNO']

c_symptom = 'Depression Joint Hyperten Diabetes Stomach Osteopor Heart Bowel Comorb_other2 Comorb_other3 num_comorbs sensory2 motor2 reflexes2 neuro_any pseudclod3 painrad_any SLR FemTens SLR_FemTens HernLoc2 HernLv3 HernTyp sps_central sps_latrec sps_neuro SpsLv1 SpsLv2 SpsLv3 SpsLv4 stenosis_worst mod_sev_levels2 DSLevel instab'
c_symptom = c_symptom.split(' ')

c_treats = [f'Treats{i+1}' for i in range(12)] + ['Meds'] + ['MedsOf{:02d}'.format(
    i+1) for i in range(10)] + [f'Prvdrs{i+1}'.format(i) for i in range(17)]
c_treats += ['b_prim_back', 'b_any_opioid',
             'b_Antidepressants', 'b_PT', 'b_Injection', 'b_NSAID']

c_outcome = 'bp pf vt mh sf rp re hp pcs mcs odi SciaticaFreq SciaticaBother back_pain_bother_n leg_pain_bother_n numb_leg_bother_n weak_leg_bother_n walking_bother_n sitting_bother_n back_pain_freq_n leg_pain_freq_n numb_leg_freq_n weak_leg_freq_n walking_freq_n sitting_freq_n vas eq5d eq5dus satis satis2_c'.split(
    ' ')

# feat_cols = c_basic_info + c_self_eval + c_symptom + c_treats + c_outcome

# 20
feat_cols_20 = ['dx', 'age', 'gender', 'racenew_4grp', 'educ', 'bmi', 'smoke',
             'Income2', 'work4grp2', 'insurance', 'Episode_fixd', 'WorkLift',
             'GuessSSU', 'GuessSNO', 'SLR', 'HernLoc2', 'HernTyp', 'pf', 'hp',
             'satis']

# 30
feat_cols_30 = ['dx', 'age', 'educ', 'educ2', 'bmi', 'smoke', 'Income2',
       'work4grp2', 'insurance', 'Any_Comp', 'treat_pref_ba',
       'better_worse_ba', 'Episode_fixd', 'WorkLift', 'GuessSSU',
       'GuessSNO', 'Joint', 'Hyperten', 'neuro_any', 'pseudclod3', 'SLR',
       'HernLoc2', 'HernLv3', 'mod_sev_levels2', 'instab', 'Treats2',
       'MedsOf02', 'MedsOf09', 'Prvdrs7', 'Prvdrs9']

# 40
feat_cols_40 = ['dx', 'age', 'educ', 'educ2', 'bmi', 'smoke', 'work4grp2',
       'insurance', 'treat_pref_ba', 'better_worse_ba', 'episode4',
       'WorkLift', 'WorkMiss', 'GuessSNO', 'Bowel', 'reflexes2',
       'pseudclod3', 'SLR', 'FemTens', 'HernLoc2', 'HernLv3', 'HernTyp',
       'SpsLv2', 'stenosis_worst', 'mod_sev_levels2', 'instab', 'Treats4',
       'Treats6', 'MedsOf01', 'MedsOf07', 'MedsOf08', 'MedsOf09',
       'MedsOf10', 'Prvdrs3', 'Prvdrs4', 'Prvdrs9', 'Prvdrs10',
       'b_prim_back', 'pf', 'mcs']

# 50
feat_cols_50 = ['age', 'gender', 'educ', 'educ2', 'marstat2', 'Income2',
       'work4grp2', 'insurance', 'Any_Comp', 'treat_pref_ba',
       'better_worse_ba', 'Episode_fixd', 'WorkLift', 'WorkMiss',
       'GuessSSU', 'GuessSNO', 'Diabetes', 'Bowel', 'num_comorbs',
       'sensory2', 'motor2', 'reflexes2', 'pseudclod3', 'SLR_FemTens',
       'HernLv3', 'HernTyp', 'SpsLv2', 'stenosis_worst',
       'mod_sev_levels2', 'DSLevel', 'instab', 'Treats4', 'Treats8',
       'MedsOf01', 'MedsOf03', 'MedsOf04', 'MedsOf05', 'MedsOf07',
       'MedsOf08', 'MedsOf09', 'MedsOf10', 'Prvdrs10', 'Prvdrs13',
       'Prvdrs14', 'b_prim_back', 'bp', 'pf', 'hp', 'back_pain_bother_n',
       'weak_leg_bother_n']

# 60
feat_cols_60 = ['dx', 'age', 'racenew_4grp', 'educ', 'educ2', 'marstat2', 'bmi',
       'smoke', 'Income2', 'work4grp2', 'insurance', 'Any_Comp',
       'treat_pref_ba', 'better_worse_ba', 'episode4', 'Episode_fixd',
       'WorkLift', 'GuessSSU', 'GuessSNO', 'Depression', 'Diabetes',
       'Stomach', 'Bowel', 'num_comorbs', 'sensory2', 'motor2',
       'reflexes2', 'pseudclod3', 'painrad_any', 'SLR', 'HernLoc2',
       'HernLv3', 'HernTyp', 'SpsLv1', 'SpsLv2', 'stenosis_worst',
       'mod_sev_levels2', 'DSLevel', 'instab', 'Treats6', 'Treats10',
       'MedsOf01', 'MedsOf02', 'MedsOf03', 'MedsOf04', 'MedsOf05',
       'MedsOf06', 'MedsOf08', 'MedsOf09', 'MedsOf10', 'Prvdrs1',
       'Prvdrs5', 'Prvdrs7', 'Prvdrs8', 'b_Antidepressants', 'b_PT',
       'b_Injection', 'pf', 'rp', 'weak_leg_freq_n']

# 70
feat_cols_70 = ['dx', 'age', 'racenew_4grp', 'educ', 'educ2', 'marstat2', 'bmi',
       'smoke', 'work4grp2', 'insurance', 'treat_pref_ba',
       'better_worse_ba', 'Episode_fixd', 'WorkLift', 'WorkMiss',
       'GuessSSU', 'GuessSNO', 'Joint', 'Diabetes', 'Osteopor',
       'Comorb_other3', 'sensory2', 'motor2', 'reflexes2', 'pseudclod3',
       'SLR', 'FemTens', 'HernLoc2', 'HernLv3', 'HernTyp', 'sps_central',
       'sps_neuro', 'SpsLv1', 'stenosis_worst', 'mod_sev_levels2',
       'instab', 'Treats1', 'Treats2', 'Treats3', 'Treats6', 'Treats8',
       'Treats10', 'MedsOf01', 'MedsOf02', 'MedsOf04', 'MedsOf05',
       'MedsOf06', 'MedsOf07', 'MedsOf08', 'MedsOf09', 'MedsOf10',
       'Prvdrs3', 'Prvdrs4', 'Prvdrs5', 'Prvdrs8', 'Prvdrs9', 'Prvdrs10',
       'Prvdrs12', 'Prvdrs13', 'Prvdrs14', 'b_any_opioid', 'b_NSAID',
       'pf', 'vt', 'mh', 're', 'pcs', 'mcs', 'leg_pain_bother_n',
       'leg_pain_freq_n']

# all
feat_cols_all = c_basic_info + c_self_eval + c_symptom + c_treats + c_outcome


# for lightgbm
# feat_cols = ['bmi', 'pcs', 'mcs', 'odi', 'vas', 'age', 'hp', 'vt', 'eq5d', 'SciaticaFreq', 'SciaticaBother', 'mh', 'pf', 'bp', 'numb_leg_bother_n', 'back_pain_freq_n', 'numb_leg_freq_n', 'back_pain_bother_n', 'weak_leg_bother_n', 'weak_leg_freq_n']

# feat_cols = ['Episode_fixd', 'SLR', 'pf', 'smoke', 'insurance', 'WorkLift', 'FemTens', 'age', 'vt', 'DSLevel', 'b_prim_back', 'work4grp2', 'mod_sev_levels2', 'educ', 'Income2', 'bmi', 'educ2', 'HernTyp', 'HernLoc2', 'GuessSNO', 'episode4', 'better_worse_ba', 'treat_pref_ba', 'satis', 'gender', 'GuessSSU', 'dx', 'racenew_4grp', 'hp', 'HernLv3']

feasible_bp = [0.,  10.,  12.,  20.,  22.,  25.,  30.,  32.,  35.,  38.,  40., 42.,  45., 48.,
               50.,  52.,  55.,  58.,  60.,  62.,  65.,  68., 70.,  75.,  78.,  80.,  88.,  90., 100.]
feasible_pf = [0.,   5.,   6.,   7.,   8.,  10.,  11.,  12.,  14.,  15.,  17., 19.,  20.,  21.,  22.,  25.,  28.,  29.,  30.,  31.,  33.,  35., 36.,  38.,  39.,  40.,  43.,  44.,
               45.,  50.,  55.,  56.,  57., 58.,  60.,  61.,  62.,  64.,  65.,  67.,  69.,  70.,  71.,  72., 75.,  78.,  79.,  80.,  81.,  83.,  85.,  88.,  89.,  90.,  92., 93.,  94.,  95., 100.]
