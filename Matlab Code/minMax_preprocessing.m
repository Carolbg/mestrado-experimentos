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
    
    imgFiltered = medfilt2(img);

    B = maxk(imgFiltered(:),100);
    meanTop10 = mean(B);

    B = mink(imgFiltered(:),100);
    meanBottom10 = mean(B);

    minMaxImg = (imgFiltered - meanBottom10)/(meanTop10-meanBottom10);
    
    figure;
    subplot(1,2,1)
    imagesc(minMaxImg);
    title('antes')

    subplot(1,2,2)
    histogram(minMaxImg);
    disp('no default 1')
    min(minMaxImg(:))
    max(minMaxImg(:))
%     
%     minMaxImg(minMaxImg < 0) = 0;
%     minMaxImg(minMaxImg > 1) = 1;
%     
%     figure;
%     subplot(1,2,1)
%     imagesc(minMaxImg);
%     title('dp')
% 
%     subplot(1,2,2)
%     histogram(minMaxImg);
%     disp('no default 2')
%     min(minMaxImg(:))
%     max(minMaxImg(:))
%     
    folderSaudaveis = strcat('saudaveis/', nomeSaudaveis(i, :), '.png');
    saveas(gcf, folderSaudaveis)
    
    %Saving original image
    numpyMinMax = py.numpy.array(minMaxImg);
    folderSaudaveis = strcat('../../Imagens_numpy_minMax_double/0Saudaveis/', nomeSaudaveis(i, :));
    py.numpy.save(folderSaudaveis, numpyMinMax);
    
     dataAugment2DImage(minMaxImg, nomeSaudaveis, i, 2, 'saudaveis/', '0Saudaveis','Imagens_numpy_minMax_double')
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
    
    imgFiltered = medfilt2(img);

    B = maxk(imgFiltered(:),100);
    meanTop10 = mean(B);

    B = mink(imgFiltered(:),100);
    meanBottom10 = mean(B);

    minMaxImg = (imgFiltered - meanBottom10)/(meanTop10-meanBottom10);
    
    
    folderDoentes = strcat('doentes/', nomeDoentes(i, :), '.png');
    saveas(gcf, folderDoentes)
    
    numpyMinMax = py.numpy.array(minMaxImg);
    folderDoentes = strcat('../../Imagens_numpy_minMax_double/1Doentes/', nomeDoentes(i, :));
    py.numpy.save(folderDoentes, numpyMinMax);
    
    dataAugment2DImage(minMaxImg, nomeSaudaveis, i, 2, 'doentes/', '1Doentes', 'Imagens_numpy_minMax_double')
    
    
    close all
end

