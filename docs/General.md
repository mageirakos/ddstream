# General Notes:

- initDBSCAN data must be accessible by the containers. Now under `/shared_data/data/.` In the future have a different permanent shared folder? Or is the /shared_data one good enough?
    [ ] Action needed? 
    -  init data parsed before streaming starts (from inside container). Streaming data sent through kafka.
[ ] InitDBSCAN has hardcoded init toy -> change to more generic version

# Datasets
[ ] They need to be normalized etc. ( check paper ) in order for the parameters minPts, Epsilon etc. to make sence

# Optimizations
1. Change everything from list to numpy array and do multiplications etc. from the numpy package
    - [ ] Complete
2. 