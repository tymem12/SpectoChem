class StandarizerSingletonLambda:
    _instance = None
    
    # stored values (default: None)
    mean_lambda = None
    std_lambda = None
    standarize = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def set_values(cls, mean_lambda, std_lambda):
        cls.mean_lambda = mean_lambda
        cls.std_lambda = std_lambda
        cls.standarize = True

    @classmethod
    def get_values(cls):
        return {
            "standarize": cls.standarize,
            "mean_lambda": cls.mean_lambda,
            "std_lambda": cls.std_lambda,
        }



class StandarizerSingletonF:
    _instance = None
    
    # stored values (default: None)
    mean_f = None
    std_f = None
    standarize = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def set_values(cls, mean_f, std_f):
        cls.mean_f = mean_f
        cls.std_f = std_f
        cls.standarize = True

    @classmethod
    def get_values(cls):
        return {
            "standarize": cls.standarize,
            "mean_f": cls.mean_f,
            "std_f": cls.std_f,
        }
