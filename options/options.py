import yaml

class Options():
    """
    Options initializer, loading options from a yaml file. 
    
    Can add additional arguments or override with kwargs
    ----------
    filepath: string of the file location
    """
    def __init__(self, **kwargs):
        with open("options/opts.yaml", "rb") as file:
            opts_dict = yaml.load(file, Loader = yaml.FullLoader)
        opts_dict.update(**kwargs)
        for option, value in opts_dict.items():
            setattr(self, option, value)
            
    