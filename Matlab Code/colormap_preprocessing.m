%% SCRIPT 1: GERAR IMAGENS COM JET COLORMAP E SALVAR COMO NUMPY ARRAY

%% Images definitions
nomeSaudaveis=[
%     'T0174.1.1.S.2013-03-20.00',
%     'T0174.1.2.S.2013-03-20.00',
%     'T0174.1.3.S.2013-03-20.00',
%     'T0174.1.4.S.2013-03-20.00',
%     'T0174.1.5.S.2013-03-20.00',
    
    'T0177.1.1.S.2013-03-20.00',
    'T0177.1.2.S.2013-03-20.00',
    'T0177.1.3.S.2013-03-20.00',
    'T0177.1.4.S.2013-03-20.00',
    'T0177.1.5.S.2013-03-20.00',
    
%     'T0182.1.1.S.2013-05-24.00',
%     'T0182.1.2.S.2013-05-24.00',
%     'T0182.1.3.S.2013-05-24.00',
%     'T0182.1.4.S.2013-05-24.00',
%     'T0182.1.5.S.2013-05-24.00',
    
    'T0188.1.1.S.2013-08-12.00',
    'T0188.1.2.S.2013-08-12.00',
    'T0188.1.3.S.2013-08-12.00',
    'T0188.1.4.S.2013-08-12.00',
    'T0188.1.5.S.2013-08-12.00',
    
    'T0189.1.1.S.2013-09-30.00',
    'T0189.1.2.S.2013-09-30.00',
    'T0189.1.3.S.2013-09-30.00',
    'T0189.1.4.S.2013-09-30.00',
    'T0189.1.5.S.2013-09-30.00',
    
    'T0190.1.1.S.2013-09-30.00',
    'T0190.1.2.S.2013-09-30.00',
    'T0190.1.3.S.2013-09-30.00',
    'T0190.1.4.S.2013-09-30.00',
    'T0190.1.5.S.2013-09-30.00',
    
    'T0191.1.1.S.2013-09-06.00',
    'T0191.1.2.S.2013-09-06.00',
    'T0191.1.3.S.2013-09-06.00',
    'T0191.1.4.S.2013-09-06.00',
    'T0191.1.5.S.2013-09-06.00',
    
    'T0193.1.1.S.2013-10-02.00',
    'T0193.1.2.S.2013-10-02.00',
    'T0193.1.3.S.2013-10-02.00',
    'T0193.1.4.S.2013-10-02.00',
    'T0193.1.5.S.2013-10-02.00',
    
    'T0194.1.1.S.2013-10-02.00',
    'T0194.1.2.S.2013-10-02.00',
    'T0194.1.3.S.2013-10-02.00',
    'T0194.1.4.S.2013-10-02.00',
    'T0194.1.5.S.2013-10-02.00',
    
    'T0195.1.1.S.2013-10-07.00',
    'T0195.1.2.S.2013-10-07.00',
    'T0195.1.3.S.2013-10-07.00',
    'T0195.1.4.S.2013-10-07.00',
    'T0195.1.5.S.2013-10-07.00',
    
    'T0196.1.1.S.2013-10-07.00',
    'T0196.1.2.S.2013-10-07.00',
    'T0196.1.3.S.2013-10-07.00',
    'T0196.1.4.S.2013-10-07.00',
    'T0196.1.5.S.2013-10-07.00',
    
    'T0199.1.1.S.2013-10-07.00',
    'T0199.1.2.S.2013-10-07.00',
    'T0199.1.3.S.2013-10-07.00',
    'T0199.1.4.S.2013-10-07.00',
    'T0199.1.5.S.2013-10-07.00',
    
    'T0200.1.1.S.2013-10-07.00',
    'T0200.1.2.S.2013-10-07.00',
    'T0200.1.3.S.2013-10-07.00',
    'T0200.1.4.S.2013-10-07.00',
    'T0200.1.5.S.2013-10-07.00',
    
    'T0201.1.1.S.2013-10-28.00',
    'T0201.1.2.S.2013-10-28.00',
    'T0201.1.3.S.2013-10-28.00',
    'T0201.1.4.S.2013-10-28.00',
    'T0201.1.5.S.2013-10-28.00',
    
    'T0208.1.1.S.2013-10-28.00',
    'T0208.1.2.S.2013-10-28.00',
    'T0208.1.3.S.2013-10-28.00',
    'T0208.1.4.S.2013-10-28.00',
    'T0208.1.5.S.2013-10-28.00',
    
    'T0211.1.1.S.2013-11-08.00',
    'T0211.1.2.S.2013-11-08.00',
    'T0211.1.3.S.2013-11-08.00',
    'T0211.1.4.S.2013-11-08.00',
    'T0211.1.5.S.2013-11-08.00',
    
    'T0212.1.1.S.2013-11-08.00',
    'T0212.1.2.S.2013-11-08.00',
    'T0212.1.3.S.2013-11-08.00',
    'T0212.1.4.S.2013-11-08.00',
    'T0212.1.5.S.2013-11-08.00',
    
    'T0216.1.1.S.2013-11-11.00',
    'T0216.1.2.S.2013-11-11.00',
    'T0216.1.3.S.2013-11-11.00',
    'T0216.1.4.S.2013-11-11.00',
    'T0216.1.5.S.2013-11-11.00',
    
    'T0217.1.1.S.2013-11-11.00',
    'T0217.1.2.S.2013-11-11.00',
    'T0217.1.3.S.2013-11-11.00',
    'T0217.1.4.S.2013-11-11.00',
    'T0217.1.5.S.2013-11-11.00',
    
    'T0218.1.1.S.2013-11-11.00',
    'T0218.1.2.S.2013-11-11.00',
    'T0218.1.3.S.2013-11-11.00',
    'T0218.1.4.S.2013-11-11.00',
    'T0218.1.5.S.2013-11-11.00',
    
    'T0219.1.1.S.2013-11-11.00',
    'T0219.1.2.S.2013-11-11.00',
    'T0219.1.3.S.2013-11-11.00',
    'T0219.1.4.S.2013-11-11.00',
    'T0219.1.5.S.2013-11-11.00',
    
    'T0220.1.1.S.2013-11-18.00',
    'T0220.1.2.S.2013-11-18.00',
    'T0220.1.3.S.2013-11-18.00',
    'T0220.1.4.S.2013-11-18.00',
    'T0220.1.5.S.2013-11-18.00',
    
    'T0221.1.1.S.2013-11-18.00',
    'T0221.1.2.S.2013-11-18.00',
    'T0221.1.3.S.2013-11-18.00',
    'T0221.1.4.S.2013-11-18.00',
    
    'T0222.1.1.S.2013-11-18.00',
    'T0222.1.2.S.2013-11-18.00',
    'T0222.1.3.S.2013-11-18.00',
    'T0222.1.4.S.2013-11-18.00',
    'T0222.1.5.S.2013-11-18.00',
    
    'T0224.1.1.S.2013-11-18.00',
    'T0224.1.2.S.2013-11-18.00',
    'T0224.1.3.S.2013-11-18.00',
    'T0224.1.4.S.2013-11-18.00',
    'T0224.1.5.S.2013-11-18.00',
    
    'T0225.1.1.S.2013-11-18.00',
    'T0225.1.2.S.2013-11-18.00',
    'T0225.1.3.S.2013-11-18.00',
    'T0225.1.4.S.2013-11-18.00',
    'T0225.1.5.S.2013-11-18.00',
    
    'T0226.1.1.S.2013-11-18.00',
    'T0226.1.2.S.2013-11-18.00',
    'T0226.1.3.S.2013-11-18.00',
    'T0226.1.4.S.2013-11-18.00',
    'T0226.1.5.S.2013-11-18.00',
    
    'T0233.1.1.S.2013-12-11.00',
    'T0233.1.2.S.2013-12-11.00',
    'T0233.1.3.S.2013-12-11.00',
    'T0233.1.4.S.2013-12-11.00',
    'T0233.1.5.S.2013-12-11.00',
    
    'T0234.1.1.S.2013-12-11.00',
    'T0234.1.2.S.2013-12-11.00',
    'T0234.1.3.S.2013-12-11.00', 
    'T0234.1.4.S.2013-12-11.00',
    'T0234.1.5.S.2013-12-11.00',
    
    'T0236.1.1.S.2014-05-21.00',
    'T0236.1.2.S.2014-05-21.00',
    'T0236.1.3.S.2014-05-21.00',
    'T0236.1.4.S.2014-05-21.00',
    'T0236.1.5.S.2014-05-21.00',
    
    'T0237.1.1.S.2014-05-21.00',
    'T0237.1.2.S.2014-05-21.00',
    'T0237.1.3.S.2014-05-21.00',
    'T0237.1.4.S.2014-05-21.00',
    'T0237.1.5.S.2014-05-21.00',
    
    'T0238.1.1.S.2014-05-21.00',
    'T0238.1.2.S.2014-05-21.00',
    'T0238.1.3.S.2014-05-21.00',
    'T0238.1.4.S.2014-05-21.00',
    'T0238.1.5.S.2014-05-21.00',
    
    'T0239.1.1.S.2014-05-26.00',
    'T0239.1.2.S.2014-05-26.00',
    'T0239.1.3.S.2014-05-26.00',
    'T0239.1.4.S.2014-05-26.00',
    'T0239.1.5.S.2014-05-26.00',
    
    'T0243.1.1.S.2014-04-15.00', 
    'T0243.1.2.S.2014-04-15.00',
    'T0243.1.3.S.2014-04-15.00',
    'T0243.1.4.S.2014-04-15.00',
    'T0243.1.5.S.2014-04-15.00',
    
    'T0244.1.1.S.2014-05-06.00',
    'T0244.1.2.S.2014-05-06.00',
    'T0244.1.3.S.2014-05-06.00',
    'T0244.1.4.S.2014-05-06.00',
    'T0244.1.5.S.2014-05-06.00',
    
    'T0259.1.1.S.2014-11-07.00', 
    'T0259.1.2.S.2014-11-07.00',
    'T0259.1.3.S.2014-11-07.00',
    'T0259.1.4.S.2014-11-07.00',
    'T0259.1.5.S.2014-11-07.00',
    
    'T0261.1.1.S.2014-11-11.00',
    'T0261.1.2.S.2014-11-11.00',
    'T0261.1.3.S.2014-11-11.00',
    'T0261.1.4.S.2014-11-11.00',
    'T0261.1.5.S.2014-11-11.00',
    
    'T0272.1.1.S.2015-03-13.00',
    'T0272.1.2.S.2015-03-13.00',
    'T0272.1.3.S.2015-03-13.00',
    'T0272.1.5.S.2015-03-13.00',
    
    'T0275.1.1.S.2015-03-13.00',
    'T0275.1.2.S.2015-03-13.00',
    'T0275.1.3.S.2015-03-13.00',
    'T0275.1.4.S.2015-03-13.00',
    'T0275.1.5.S.2015-03-13.00',
    
    'T0276.1.1.S.2015-03-13.00',
    'T0276.1.2.S.2015-03-13.00',
    'T0276.1.3.S.2015-03-13.00',
    'T0276.1.4.S.2015-03-13.00',
    'T0276.1.5.S.2015-03-13.00',
];

nomeDoentes=[
    'T0138.2.1.S.2013-09-06.00',
    'T0138.2.2.S.2013-09-06.00',
    'T0138.2.3.S.2013-09-06.00',
    'T0138.2.4.S.2013-09-06.00',
    'T0138.2.5.S.2013-09-06.00', 
    
    'T0179.1.1.S.2013-08-16.00',
    'T0179.1.2.S.2013-08-16.00',
    'T0179.1.3.S.2013-08-16.00',
    'T0179.1.4.S.2013-08-16.00',
    'T0179.1.5.S.2013-08-16.00',
    
    'T0180.1.1.S.2013-08-16.00',
    'T0180.1.2.S.2013-08-16.00',
    'T0180.1.3.S.2013-08-16.00',
    'T0180.1.4.S.2013-08-16.00',
    'T0180.1.5.S.2013-08-16.00',
    
    'T0181.1.1.S.2013-08-16.00',
    'T0181.1.2.S.2013-08-16.00',
    'T0181.1.3.S.2013-08-16.00',
    'T0181.1.4.S.2013-08-16.00',
    'T0181.1.5.S.2013-08-16.00',
    
    'T0192.1.1.S.2013-09-06.00',
    'T0192.1.2.S.2013-09-06.00',
    'T0192.1.3.S.2013-09-06.00',
    'T0192.1.4.S.2013-09-06.00',
    'T0192.1.5.S.2013-09-06.00',
    
    'T0198.2.1.S.2014-11-11.00',
    'T0198.2.2.S.2014-11-11.00',
    'T0198.2.3.S.2014-11-11.00',
    'T0198.2.4.S.2014-11-11.00',
    'T0198.2.5.S.2014-11-11.00',
    'T0202.1.1.S.2013-10-11.00',
    'T0202.1.2.S.2013-10-11.00',
    'T0202.1.3.S.2013-10-11.00',
    'T0202.1.4.S.2013-10-11.00',
    'T0202.1.5.S.2013-10-11.00',
    'T0203.1.1.S.2013-10-11.00',
    'T0203.1.2.S.2013-10-11.00',
    'T0203.1.3.S.2013-10-11.00',
    'T0203.1.4.S.2013-10-11.00',
    'T0203.1.5.S.2013-10-11.00',
    
    'T0204.1.1.S.2013-10-11.00', 
    'T0204.1.2.S.2013-10-11.00',
    'T0204.1.3.S.2013-10-11.00',
    'T0204.1.4.S.2013-10-11.00',
    'T0204.1.5.S.2013-10-11.00',
    
    'T0209.1.1.S.2013-11-08.00',
    'T0209.1.2.S.2013-11-08.00',
    'T0209.1.3.S.2013-11-08.00',
    'T0209.1.4.S.2013-11-08.00',
    'T0209.1.5.S.2013-11-08.00',
    
    'T0210.1.1.S.2013-11-08.00',
    'T0210.1.2.S.2013-11-08.00',
    'T0210.1.3.S.2013-11-08.00',
    'T0210.1.4.S.2013-11-08.00',
    'T0210.1.5.S.2013-11-08.00',
    
    'T0213.1.1.S.2013-11-08.00', 
    'T0213.1.2.S.2013-11-08.00',
    'T0213.1.3.S.2013-11-08.00',
    'T0213.1.4.S.2013-11-08.00',
    'T0213.1.5.S.2013-11-08.00',
    
    'T0240.1.1.S.2014-07-18.00',
    'T0240.1.2.S.2014-07-18.00',
    'T0240.1.3.S.2014-07-18.00',
    'T0240.1.4.S.2014-07-18.00',
    'T0240.1.5.S.2014-07-18.00',
    
    'T0241.1.1.S.2014-07-18.00',
    'T0241.1.2.S.2014-07-18.00',
    'T0241.1.3.S.2014-07-18.00',
    'T0241.1.4.S.2014-07-18.00',
    'T0241.1.5.S.2014-07-18.00',
    
    'T0245.1.1.S.2014-08-22.00',
    'T0245.1.2.S.2014-08-22.00',
    'T0245.1.3.S.2014-08-22.00',
    'T0245.1.4.S.2014-08-22.00',
    'T0245.1.5.S.2014-08-22.00',
    
    'T0246.1.1.S.2014-08-22.00',
    'T0246.1.2.S.2014-08-22.00',
    'T0246.1.3.S.2014-08-22.00',
    'T0246.1.4.S.2014-08-22.00',
    'T0246.1.5.S.2014-08-22.00',
    
    'T0255.1.1.S.2014-08-22.00', 
    'T0255.1.2.S.2014-08-22.00',
    'T0255.1.3.S.2014-08-22.00',
    'T0255.1.4.S.2014-08-22.00',
    'T0255.1.5.S.2014-08-22.00',
    
    'T0256.1.1.S.2014-10-10.00',
    'T0256.1.2.S.2014-10-10.00',
    'T0256.1.3.S.2014-10-10.00',
    'T0256.1.4.S.2014-10-10.00',
    'T0256.1.5.S.2014-10-10.00',
    
    'T0257.1.1.S.2014-10-17.00',
    'T0257.1.2.S.2014-10-17.00',
    'T0257.1.3.S.2014-10-17.00',
    'T0257.1.4.S.2014-10-17.00',
    'T0257.1.5.S.2014-10-17.00',
    
    'T0258.1.1.S.2014-10-17.00',
    'T0258.1.3.S.2014-10-17.00',
    'T0258.1.4.S.2014-10-17.00',
    'T0258.1.5.S.2014-10-17.00',
    'T0260.1.1.S.2014-11-11.00',
    'T0260.1.2.S.2014-11-11.00',
    'T0260.1.3.S.2014-11-11.00',
    'T0260.1.4.S.2014-11-11.00',
    'T0260.1.5.S.2014-11-11.00',
    'T0263.1.1.S.2014-12-12.00',
    'T0263.1.2.S.2014-12-12.00',
    'T0263.1.3.S.2014-12-12.00',
    'T0263.1.4.S.2014-12-12.00',
    'T0263.1.5.S.2014-12-12.00',
    'T0264.1.1.S.2015-01-16.00', 
    'T0264.1.2.S.2015-01-16.00',
    'T0264.1.3.S.2015-01-16.00',
    'T0264.1.4.S.2015-01-16.00',
    'T0264.1.5.S.2015-01-16.00',
    'T0266.1.1.S.2015-01-16.00',
    'T0266.1.2.S.2015-01-16.00',
    'T0266.1.3.S.2015-01-16.00',
    'T0266.1.4.S.2015-01-16.00',
    'T0266.1.5.S.2015-01-16.00',
    'T0267.1.1.S.2015-01-16.00',
    'T0267.1.2.S.2015-01-16.00',
    'T0267.1.3.S.2015-01-16.00',
    'T0267.1.4.S.2015-01-16.00',
    'T0267.1.5.S.2015-01-16.00',
    'T0268.1.1.S.2015-01-23.00', 
    'T0268.1.2.S.2015-01-23.00',
    'T0268.1.3.S.2015-01-23.00',
    'T0268.1.4.S.2015-01-23.00',
    'T0269.1.1.S.2015-01-23.00',
    'T0269.1.2.S.2015-01-23.00',
    'T0269.1.3.S.2015-01-23.00',
    'T0269.1.4.S.2015-01-23.00',
    'T0269.1.5.S.2015-01-23.00',
    'T0270.1.1.S.2015-01-30.00',
    'T0270.1.2.S.2015-01-30.00',
    'T0270.1.3.S.2015-01-30.00',
    'T0270.1.4.S.2015-01-30.00',
    'T0270.1.5.S.2015-01-30.00',
    'T0271.1.1.S.2015-01-27.00',
    'T0271.1.2.S.2015-01-27.00', 
    'T0271.1.3.S.2015-01-27.00',
    'T0271.1.4.S.2015-01-27.00',
    'T0271.1.5.S.2015-01-27.00',
    'T0273.1.1.S.2015-03-13.00',
    'T0273.1.2.S.2015-03-13.00',
    'T0273.1.3.S.2015-03-13.00',
    'T0273.1.4.S.2015-03-13.00',
    'T0273.1.5.S.2015-03-13.00',
    'T0277.1.1.S.2015-03-20.00',
    'T0277.1.2.S.2015-03-20.00',
    'T0277.1.3.S.2015-03-20.00',
    'T0277.1.4.S.2015-03-20.00',
    'T0277.1.5.S.2015-03-20.00',
    'T0278.1.1.S.2015-03-20.00',
    'T0278.1.2.S.2015-03-20.00', 
    'T0278.1.3.S.2015-03-20.00',
    'T0278.1.4.S.2015-03-20.00',
    'T0278.1.5.S.2015-03-20.00',
    'T0281.1.1.S.2015-05-22.00',
    'T0281.1.2.S.2015-05-22.00',
    'T0281.1.3.S.2015-05-22.00',
    'T0281.1.4.S.2015-05-22.00',
    'T0281.1.5.S.2015-05-22.00',
    'T0282.1.1.S.2015-07-20.00',
    'T0282.1.2.S.2015-07-20.00',
    'T0282.1.3.S.2015-07-20.00',
    'T0282.1.4.S.2015-07-20.00',
    'T0282.1.5.S.2015-07-20.00',
    'T0283.1.1.S.2015-05-22.00',
    'T0283.1.2.S.2015-05-22.00', 
    'T0283.1.3.S.2015-05-22.00',
    'T0283.1.4.S.2015-05-22.00',
    'T0283.1.5.S.2015-05-22.00',
    'T0285.1.1.S.2015-07-20.00',
    'T0285.1.2.S.2015-07-20.00',
    'T0285.1.3.S.2015-07-20.00',
    'T0285.1.4.S.2015-07-20.00',
    'T0285.1.5.S.2015-07-20.00',
    'T0286.1.1.S.2015-05-22.00',
    'T0286.1.2.S.2015-05-22.00',
    'T0286.1.3.S.2015-05-22.00',
    'T0286.1.4.S.2015-05-22.00',
    'T0286.1.5.S.2015-05-22.00',
    'T0287.1.1.S.2015-07-20.00',
    'T0287.1.2.S.2015-07-20.00', 
    'T0287.1.3.S.2015-07-20.00',
    'T0287.1.4.S.2015-07-20.00',
    'T0287.1.5.S.2015-07-20.00'
];

%% Leitura imagens saudaveis

saudaveis = '../../Imagens_TXT_Estaticas_Balanceadas_allData/0Saudavel/';
sizeSaudaveis = size(nomeSaudaveis,1);
pSaudaveis = cell(sizeSaudaveis,1);
pSaudaveisRGB = cell(sizeSaudaveis,1);
minSaudaveis = ones(sizeSaudaveis);
maxSaudaveis = ones(sizeSaudaveis);

for i = 1:sizeSaudaveis
    fullPath = strcat(saudaveis, nomeSaudaveis(i, :), '.txt');
    img = load(fullPath);
    pSaudaveis{i} = img;
    
    % Essa parte aqui que trata a conversao pra RBG, das linhas 455 ate a
    % 468
    f = figure;
    cmap = colormap(f,jet);
    h = imagesc(img);
    Cdata = h.CData;
    cmap = colormap;

    % make it into a index image.
    cmin = min(Cdata(:));
    cmax = max(Cdata(:));
    m = length(cmap);
    
    index = fix((Cdata-cmin)/(cmax-cmin)*m)+1;
    % Then to RGB
    RGB = ind2rgb(index, cmap);
    
    minSaudaveis(i) = min(RGB(:));
    maxSaudaveis(i) = max(RGB(:));
    pSaudaveisRGB{i} = RGB;
    
    figure;
    subplot(1,2,1)
    imagesc(RGB);
    title(nomeSaudaveis(i, :))

    subplot(1,2,2)
    histogram(RGB);
  
    folderSaudaveis = strcat('saudaveis/', nomeSaudaveis(i, :), '.png');
    saveas(gcf, folderSaudaveis)
    
    %Saving original image
    numpyRGB = py.numpy.array(RGB);
    folderSaudaveis = strcat('../../Imagens_numpy_array_allData_asMinMax_double/0Saudaveis/', nomeSaudaveis(i, :));
    py.numpy.save(folderSaudaveis, numpyRGB);
    
     dataAugment(img, RGB, nomeSaudaveis, i, 2, 'saudaveis/', '0Saudaveis','Imagens_numpy_array_allData_asMinMax_double')
%     %getting mean
%     I = img;
%     RGBParsed = RGB;
%     thresh = multithresh(I);
%     seg_I = imquantize(I,thresh);
%     %figure; imagesc(seg_I);
%     
%     RGBParsed1 = RGBParsed(:,:,1);
%     RGBParsed2 = RGBParsed(:,:,2);
%     RGBParsed3 = RGBParsed(:,:,3);
%     meanValue1 = mean(RGBParsed1(seg_I == 1));
%     meanValue2 = mean(RGBParsed2(seg_I == 1));
%     meanValue3 = mean(RGBParsed3(seg_I == 1));
%     
%     %Generating and saving one altered image
%     imOriginal = RGB;
%     tform = randomAffine2d('Rotation',[-45 45], 'XReflection',true,'YReflection',true); 
%     outputView = affineOutputView(size(imOriginal),tform);
%     imAlterada = imwarp(imOriginal,tform,'OutputView',outputView,'FillValues',[meanValue1 meanValue2 meanValue3]);
% %     imAlteradaCor = jitterColorHSV(imAlterada,'Contrast',[1.2 1.4],'Saturation',[-0.4 -0.1],'Brightness',[-0.2 0.2]);
%     imAlteradaCor=imAlterada;
%     figure;
%     imagesc(imAlteradaCor);
%     folderSaudaveis = strcat('saudaveis/', nomeSaudaveis(i, :), '_alt1.png');
%     saveas(gcf, folderSaudaveis)
%     
%     numpyRGB = py.numpy.array(imAlteradaCor);
%     folderSaudaveis = strcat('../../Imagens_numpy_array_allData_semCores_3/0Saudaveis/', nomeSaudaveis(i, :), '_alt_1');
%     py.numpy.save(folderSaudaveis, numpyRGB);
%     
    
    %Generating and saving other altered image
%     tform = randomAffine2d('Rotation',[-45 45], 'XReflection',true,'YReflection',true); 
%     outputView = affineOutputView(size(imOriginal),tform);
%     imAlterada = imwarp(imOriginal,tform,'OutputView',outputView,'FillValues',[meanValue1 meanValue2 meanValue3]);
% %     imAlteradaCor = jitterColorHSV(imAlterada,'Contrast',[1.2 1.4],'Saturation',[-0.4 -0.1],'Brightness',[-0.2 0.2]);
%     figure;
%     imAlteradaCor=imAlterada;
%     imagesc(imAlteradaCor);
%     folderSaudaveis = strcat('saudaveis/', nomeSaudaveis(i, :), '_alt2.png');
%     saveas(gcf, folderSaudaveis)
%     
%     numpyRGB = py.numpy.array(imAlteradaCor);
%     folderSaudaveis = strcat('../../Imagens_numpy_array_allData_semCores_3/0Saudaveis/', nomeSaudaveis(i, :), '_alt_2');
%     py.numpy.save(folderSaudaveis, numpyRGB);
   
    close all
end

%% Leitura imagens doentes

doentes = '../../Imagens_TXT_Estaticas_Balanceadas_allData/1Doente/';
sizeDoentes = size(nomeDoentes,1);
pDoentes = cell(sizeDoentes,1);
pDoentesRGB = cell(sizeDoentes,1);

minDoentes = ones(sizeDoentes);
maxDoentes = ones(sizeDoentes);

for i = 1:sizeDoentes
    fullPath = strcat(doentes, nomeDoentes(i, :), '.txt');
    img = load(fullPath); 
    pDoentes{i} = img;
    
     % 
    f = figure;
    cmap = colormap(f,jet);
    h = imagesc(img);
    Cdata = h.CData;
    cmap = colormap;

    % make it into a index image.
    cmin = min(Cdata(:));
    cmax = max(Cdata(:));
    m = length(cmap);
    index = fix((Cdata-cmin)/(cmax-cmin)*m)+1;
    
    % Then to RGB
    RGB = ind2rgb(index, cmap);
    
    minDoentes(i) = min(RGB(:));
    maxDoentes(i) = max(RGB(:));
    pDoentesRGB{i} = RGB;
   
    figure;
    subplot(1,2,1)
    imagesc(RGB);
    title(nomeDoentes(i, :))

    subplot(1,2,2)
    histogram(RGB);
    
    folderDoentes = strcat('doentes/', nomeDoentes(i, :), '.png');
    saveas(gcf, folderDoentes)
    
    numpyRGB = py.numpy.array(RGB);
    folderDoentes = strcat('../../Imagens_numpy_array_allData_asMinMax_double/1Doentes/', nomeDoentes(i, :));
    py.numpy.save(folderDoentes, numpyRGB);
    
     dataAugment(img, RGB, nomeSaudaveis, i, 2, 'doentes/', '1Doentes', 'Imagens_numpy_array_allData_asMinMax_double')
    
%     %getting mean
%     I = img;
%     RGBParsed = RGB;
%     thresh = multithresh(I);
%     seg_I = imquantize(I,thresh);
%     %figure; imagesc(seg_I);
%     
%     RGBParsed1 = RGBParsed(:,:,1);
%     RGBParsed2 = RGBParsed(:,:,2);
%     RGBParsed3 = RGBParsed(:,:,3);
%     meanValue1 = mean(RGBParsed1(seg_I == 1));
%     meanValue2 = mean(RGBParsed2(seg_I == 1));
%     meanValue3 = mean(RGBParsed3(seg_I == 1));
%     
%     imOriginal = RGB;
%     tform = randomAffine2d('Rotation',[-45 45], 'XReflection',true,'YReflection',true); 
%     outputView = affineOutputView(size(imOriginal),tform);
%     imAlterada = imwarp(imOriginal,tform,'OutputView',outputView,'FillValues',[meanValue1 meanValue2 meanValue3]);
% %     imAlteradaCor = jitterColorHSV(imAlterada,'Contrast',[1.2 1.4],'Saturation',[-0.4 -0.1],'Brightness',[-0.2 0.2]);
%     imAlteradaCor = imAlterada;
%     figure;
%     imagesc(imAlteradaCor);
%     folderDoentes = strcat('doentes/', nomeDoentes(i, :), '_alt1.png');
%     saveas(gcf, folderDoentes)
%     numpyRGB = py.numpy.array(imAlteradaCor);
%     folderDoentes = strcat('../../Imagens_numpy_array_allData_semCores_3/1Doentes/', nomeDoentes(i, :), '_alt_1');
%     py.numpy.save(folderDoentes, numpyRGB);
%     
%     %Generating and saving other altered image
%     tform = randomAffine2d('Rotation',[-45 45], 'XReflection',true,'YReflection',true); 
%     outputView = affineOutputView(size(imOriginal),tform);
%     imAlterada = imwarp(imOriginal,tform,'OutputView',outputView,'FillValues',[meanValue1 meanValue2 meanValue3]);
% %     imAlteradaCor = jitterColorHSV(imAlterada,'Contrast',[1.2 1.4],'Saturation',[-0.4 -0.1],'Brightness',[-0.2 0.2]);
%     imAlteradaCor = imAlterada;
%     figure;
%     imagesc(imAlteradaCor);
%     folderDoentes = strcat('doentes/', nomeDoentes(i, :), '_alt2.png');
%     saveas(gcf, folderDoentes)
%     
%     numpyRGB = py.numpy.array(imAlteradaCor);
%     folderDoentes = strcat('../../Imagens_numpy_array_allData_semCores_3/1Doentes/', nomeDoentes(i, :), '_alt_2');
%     py.numpy.save(folderDoentes, numpyRGB);
    
    close all
end

%% Generate Mask 1
% 
% for i = 1:size(pSaudaveis, 1)
%     I = pSaudaveis{i};
%     
%     %Conversao de tipo
%     top = mean(maxk(I(:),100));
%     I = uint8((255/top)*I);
%     imagesc(I)
%     
%     [L,Centers] = imsegkmeans(I,4, 'NormalizeInput', true);
%     B = labeloverlay(I,L);
%     fig = figure;
%     imshow(B)
%     saveas(fig, 'imsegkmeans_4', 'png')
% 
%     [L,Centers] = imsegkmeans(I,5, 'NormalizeInput', true);
%     B = labeloverlay(I,L);
%     fig = figure;
%     imshow(B)
%     saveas(fig, 'imsegkmeans_5', 'png')
% 
%     [~, threshold] = edge(I, 'canny');
%     BWs = edge(I,'canny', threshold);
%     se90 = strel('line', 3, 90); 
%     se0 = strel('line', 3, 0);
%     BWsdil = imdilate(BWs, [se90 se0]);
%     BWdfill = imfill(BWsdil, 'holes'); figure, 
%     fig = imshow(BWdfill);
%     title('Preprocessed canny');
%     saveas(fig, 'canny', 'png')
% end
% 
% 
% M = ones(size(I,1), size(I,2));
% M(seg_I<=1) = 0;
% imagesc(M*I)
% 
% % t1 = size(M,1);
% % t2 = size(M,2);
% % for k = 1:t1
% %     for j = 1:t2
% %         if((x1(k,j) <=2 || x1(k,j) ==6))
% %             M(k,j) = 0;
% %         end
% %     end
% % end
% 
% %%
% for i = 1:sizeSaudaveis
%     fullPath = strcat(saudaveis, nomeSaudaveis(i, :));
%     img = load(fullPath); 
%     pSaudaveis{i} = img;
% end
