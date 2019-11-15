from skopt import BayesSearchCV
import skopt
from skopt.plots import plot_convergence, plot_regret, plot_evaluations, plot_objective
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from ..base.base import Base
from ..utils import isStr, StudyDict, merge, T, F

from typing import Dict

from collections improt defaultdict
from sklearn.pipeline import Pipeline

def to_pipeline(self,i=0,name="model"):
    return Pipeline([(name,self.models[i])])

from skopt import dump
class bestScoreCallback:
    def __init__(self,i=0):
        self.i=i
    def __call__(self,res):
        score = cv.best_score_
        print("[{}] best score: {}".format(self.i,score))
        self.i+=1

class CheckpointSaver2(object):
    """
    Save current state after each iteration with `skopt.dump`.
    Example usage:
        import skopt
        checkpoint_callback = skopt.callbacks.CheckpointSaver("./result.pkl")
        skopt.gp_minimize(obj_fun, dims, callback=[checkpoint_callback])
    Parameters
    ----------
    * `checkpoint_path`: location where checkpoint will be saved to;
    * `dump_options`: options to pass on to `skopt.dump`, like `compress=9`
    """
    def __init__(self, checkpoint_path, **dump_options):
        self.checkpoint_path = checkpoint_path
        self.dump_options = dump_options
        self.opti={}

    def __call__(self, res):
        """
        Parameters
        ----------
        * `res` [`OptimizeResult`, scipy object]:
            The optimization as a OptimizeResult object.
        """
        self.opti[res.optimizer]=res
        dump(self.opti, self.checkpoint_path, **self.dump_options)


class Tuned(Base):
	EXPORTABLE=["resultat","obj","best_estimator_","best_score_","best_params_","best_index_","scorer_","n_splits_"]
	def __init__(self,resultat=None,best_estimator_=None,best_score_=None,best_params_=None,best_index_=None,scorer_=None,n_splits_=None, obj=None,  ID=None):
		super(Tuned, self).__init__(ID=ID)
		self.resultat=resultat
		self.obj=obj
		self.best_estimator_=best_estimator_
		self.best_score_=best_score_
		self.best_params_=best_params_
		self.best_index_=best_index_
		self.scorer_=scorer_
		self.n_splits_=n_splits_


class HyperTune(Base):
	EXPORTABLE=["tuning","_namesCurr"]
	TYPES_=["random","grid","bayes","bayes_rf","bayes_gp","bayes_dummy","bayes_et","bayes_gbrt"]
	def __init__(self, tuning:Dict[str,Tuned]=None,ID=None):
		super(HyperTune, self).__init__(ID=ID)
		self.tuning=StudyDict({}) if tuning is None else tuning
		self._namesCurr=None

	@property
	def curr(self):
		return self.tuning[self._namesCurr]
	
	def tune(self,mod,hyper_params,typeOfTune="random",opts={},optsFit={}):
		"""
		typeOfTune: "random" RandomizedSearchCV, "grid" GridSearchCV, "bayes" or "bayes_gp" BayesSearchCV ["Gaussian Process"], "bayes_rf" BayesSearchCV ["Random Forest"], "bayes_dummy" BayesSearchCV ["Dummy"], "bayes_et" BayesSearchCV ["Extra Trees"], "bayes_gbrt" BayesSearchCV ["gradient boosted trees"]
		"""
		modsN=self.papa._models.mappingNamesModelsInd
		modsN2=self.papa._models.namesModels
		mod=modsN[mod] if isStr(mod) else mod
		modelName=mod if isStr(mod) else modsN2[mod]
		## TUNINSSS
		if typeOfTune not in self.TYPES_:
			raise NotImplementedError("{} not implemented yet , only {}".format(typeOfTune,self.TYPES_))
		
		model=self.papa._models.models[mod]
		X_train=self.papa.X_train
		y_train=self.papa.y_train
		if typeOfTune == "random":
			deff=dict(random_state=42,n_jobs=-1,return_train_score=True)
			opts=merge(deff,opts,add=False)
			obj=RandomizedSearchCV(model,hyper_params,**opts)
			obj.fit(X_train,y_train,**optsFit)
			resultat=obj.cv_results_
			best_estimator_=obj.best_estimator_
			best_score_=obj.best_score_
			best_index_=obj.best_index_
			scorer_=obj.scorer_
			n_splits_=obj.n_splits_
			best_params_=obj.best_params_
		elif typeOfTune == "grid":
			deff=dict(n_jobs=-1,return_train_score=True)
			opts=merge(deff,opts,add=False)
			obj=GridSearchCV(model,hyper_params,**opts)
			obj.fit(X_train,y_train,**optsFit)
			resultat=obj.cv_results_
			best_estimator_=obj.best_estimator_
			best_score_=obj.best_score_
			best_index_=obj.best_index_
			scorer_=obj.scorer_
			n_splits_=obj.n_splits_
			best_params_=obj.best_params_
		elif typeOfTune in ["bayes","bayes_gp", "bayes_dummy","bayes_et","bayes_rf"]:
			fit_=dict(deadlineStopper=False,
					  deltaYStopper=False)
			opt2=merge(fit_,optsFit,add=F)
			# fit=dict()
			if opt2.get("deadlineStopper") is not None and not opt2.get("deadlineStopper"):
				opt2.pop("deadlineStopper")
				deadtime=120
				if opt2.get("deadlineStopperTime") is not None:
					deadtime=opt2.pop("deadlineStopperTime")

				dd2=skopt.callbacks.DeadlineStopper(deadtime)
				opt2["callback"]=(opt2 if "callback" in opt2 else []) + [dd2]

			if opt2.get("deltaYStopper") is not None and not opt2.get("deltaYStopper"):
				opt2.pop("deltaYStopper")
				delta=0.01
				if opt2.get("deltaYStopperDelta") is not None:
					delta=opt.pop("deltaYStopperDelta")
				dd2=skopt.callbacks.DeltaYStopper(delta)
				opt2["callback"]=(opt2 if "callback" in opt2 else []) + [dd2]

			if opt2.get("CheckpointSaver") is not None and not opt2.get("CheckpointSaver"):
				opt2.pop("CheckpointSaver")
				path="./result.pkl"
				if opt2.get("CheckpointSaverPath") is not None:
					path=opt.pop("CheckpointSaverPath")
				dd2 = CheckpointSaver2(path)
				opt2["callback"]=(opt2 if "callback" in opt2 else []) + [dd2]



			deff=dict(n_jobs=-1,n_iter=20,
                 return_train_score=True,
                 random_state=42,progressBar=True,
                 verbose=False,cv=3)
			opts=merge(deff,opts,add=False)
			pb=False
			if "verbose" in opts and opts.get("verbose"):
				dd2=bestScoreCallback()
				opts2["callback"]=(opt2 if "callback" in opt2 else []) + [dd2]

			if "progressBar" in opts and opts.get("progressBar"):
				pb=True
				opts.pop("progressBar")


			optimizer_kwargs={'base_estimator': 'GP'}
			optimizer_kwargs["acq_optimizer"]="sampling"

			if typeOfTune=="bayes_dummy":
				optimizer_kwargs["base_estimator"]="dummy"

			else:
				 acq_optimizer_kwargs = {
			        "n_points": 10000, "n_restarts_optimizer": 5,
			        "n_jobs": -1}
			    acq_func_kwargs = {"xi": 0.01, "kappa": 1.96}
				optimizer_kwargs["acq_func_kwargs"]=acq_func_kwargs
				optimizer_kwargs["acq_optimizer_kwargs"]=acq_optimizer_kwargs

				if typeOfTune in ["bayes_gp","bayes"]:
					optimizer_kwargs["acq_func"]="gp_hedge"
					
				elif typeOfTune == "bayes_rf":
					optimizer_kwargs["acq_func"]="EI"
					optimizer_kwargs["base_estimator"]="RF"

				elif typeOfTune=="bayes_et":
					optimizer_kwargs["acq_func"]="EI"
					optimizer_kwargs["base_estimator"]="ET"
				elif typeOfTune == "bayes_gbrt":
					optimizer_kwargs["acq_func"]="EI"
					# optimizer_kwargs["acq_optimizer"]="auto"


			# if "callback" in opts2:
				# optimizer_kwargs["callback"]=opts2.pop("callback")
			opts=merge(dict(optimizer_kwargs),opts,add=False)
			cv=BayesSearchCV(model,
                 hyper_params,#optimizer_kwargs=dict(callback=[checkpoint_callback]),
                 **opts)
			nbOpts=len(cv.optimizers_)
			if pb:
				nameM="hyper-{}".format(modelName)
				dd2=ProgressBarCalled(total=cv.total_iterations,
                                   name=nameM,
                                   generatorExitAsSuccess=True,
                                  fnCalled=lambda args,xargs,pb: setattr(pb,"name",nameM+str(args[0].optimizer.id)+"/"+str(nbOpts)+"]" ))
				opts2["callback"]=(opt2 if "callback" in opt2 else []) + [dd2]
			cv.fit(X_train,
   					y_train,**opts2)
			resultat=obj.cv_results_
			best_estimator_=obj.best_estimator_
			best_score_=obj.best_score_
			best_index_=obj.best_index_
			scorer_=obj.scorer_
			n_splits_=obj.n_splits_
			best_params_=obj.best_params_

		res=Tuned()
		res.resultat=resultat
		res.obj=obj
		res.best_estimator_=best_estimator_
		res.best_score_=best_score_
		res.best_params_=best_params_
		res.best_index_=best_index_
		res.scorer_=scorer_
		res.n_splits_=n_splits_
		self.tuning[res.ID]=res
		self._namesCurr=res.ID
