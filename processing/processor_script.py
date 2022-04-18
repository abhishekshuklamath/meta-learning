import numpy as onp
import pandas as pd



#read data
ethnic_dic=pd.read_csv('sampleID.csv',usecols=['Sample (Male/Female/Unknown)','Population(s)'])
#num_rows=20000
header_line=42
#in final form get rid of num_rows and remove nrows from read.csv
data=pd.read_csv('ALL.autosomes.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz',sep='\t',
                header=header_line)
#task='MXL' #the ethnic group we want to test
removal_list=['HG00104',
 'HG00134',
 'HG00135',
 'HG00152',
 'HG00156',
 'HG00249',
 'HG00270',
 'HG00302',
 'HG00303',
 'HG00312',
 'HG00359',
 'HG00377',
 'HG01471',
 'HG02168',
 'HG02169',
 'HG02170',
 'HG02173',
 'HG02176',
 'HG02358',
 'HG02405',
 'HG02436',
 'HG03171',
 'HG03393',
 'HG03398',
 'HG03431',
 'HG03462',
 'HG03549',
 'HG04301',
 'HG04302',
 'HG04303',
 'NA18527',
 'NA18576',
 'NA18791',
 'NA18955',
 'NA19044',
 'NA19359',
 'NA19371',
 'NA19398',
 'NA20537',
 'NA20816',
 'NA20829',
 'NA20831',
 'NA20873',
 'NA20883',
 'NA21121']


#processing data
def process_dict(data):
    data=data.rename(columns={'Sample (Male/Female/Unknown)':'ID','Population(s)':'eth'})
    data['eth']=data['eth'].apply(lambda x: x.split(',')[-1][1:])
    data['ID']=data['ID'].apply(lambda x: x.split(' ')[0])
    data.index=data['ID']
    data=data.drop('ID',axis=1)
    data=data.drop('NA18498')
    return data

def process_data(data,num_causal_snps=200):
    data=data[data['ALT'].isin(['A','C','G','T'])] #select SNPs with single ALT allele
    data['INFO']=data['INFO'].apply(lambda x: float(x.split(';')[3].split('=')[-1])) #extract allele freq information
    data=data[data['INFO']>0.05] #choose SNPs with allele freq more than 0.05
    data.index=data['POS'] #set ID col as index
    data=data.drop(['ID','#CHROM','POS','REF','ALT','QUAL','FILTER','INFO','FORMAT'],axis=1) #drop columns other than individual data
    data=data.applymap(lambda x: 2 if x=='1|1' else(0 if x=='0|0' else 1)) #sets 0|0 to 0 ...
    data=data.drop(removal_list,axis=1)
    data=data.T
    causal_snps=onp.arange(0,len(data.columns),len(data.columns)//num_causal_snps)
    data=data[data.columns[causal_snps]]
    data['eth']=eth_ID['eth']
    data['ID']=data.index
    data=data.set_index(['eth','ID'])
    return data





eth_ID=process_dict(ethnic_dic)
eth=eth_ID['eth'].unique()




data1=process_data(data)
pd.to_csv('processed_data.csv')
eth_ID.to_csv('eth_ID.csv')