import pandas as pd
import plotnine as p9
import mizani

results_file = 'results_2022-12-13_Approach_summaries.csv'
header_data = pd.read_csv(results_file, nrows=1)
df = pd.read_csv(results_file, skiprows=2)
df['time'] = pd.to_datetime(df['level_0'])

(p9.ggplot(df)+
	p9.geom_point(p9.aes(x='time', y='bout_behavior', color='factor(longterm_idx)'))+
	p9.scale_x_datetime(breaks=mizani.breaks.date_breaks('6 hour'), labels=mizani.formatters.date_format('%Y-%m-%d %H-%M-%S'))+
	p9.facet_wrap('exp_prefix', scales='free_x')+
	p9.theme_bw()+
	p9.theme(axis_text_x=p9.element_text(rotation=90, hjust=0.5))+
	p9.labs(title=header_data['Behavior'][0])
).draw().show()
