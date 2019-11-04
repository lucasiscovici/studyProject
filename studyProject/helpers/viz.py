# import functools

# def catch_exception_viz(f,obj,kw="me"):
#     @functools.wraps(f)
#     def func(*args, **kwargs):
#         try:
#             return f(*args, **kwargs)
#         except Exception as e:
#             try:
#                 kwargs[kw]=obj
#                 return f(*args, **kwargs)
#             except Exception as e2:
#                 print(str(e))
#                 print(str(e2))
#                 raise e
#     return func


# class vizHelper(object):
#     def __init__(self, arg, curr=None):
#         self.__obj = arg
#         self.__curr = arg if curr is None else curr 

#     def __getattr__(self,k):
#         rep=getattr(self.__curr,k)
#         if callable(rep) and type(rep)=="function":
#             return catch_exception_viz(rep,self.__obj)
#         return vizHelper(self.__obj,rep)

#     def __dir__(self):
#         return self.__curr.__dir__()
    
#     def __repr__(self):
#         try:
#             return self.__curr.__repr__()
#         except:
#             return object.__repr__(self.__curr)

#     def __getitem__(self,k):
#         rep=self.__curr[k]
#         if callable(rep) and type(rep)=="function":
#             return catch_exception_viz(rep,self.__obj)
#         return viz_helper(self.__obj,rep)