import plotly_study
from ..utils import F, merge
from ..utils import setattrAndReturnSelf



# def add_title(self,title="Title",marginT=50):
#     return self.update_layout(title=title,
#                        margin=dict(t=marginT))

# plotly_study.graph_objs._figure.Figure.add_title=addT
# import plotly_study.graph_objs as go
# import copy as _copy
# def __init3__(
#         self,
#         arg=None,
#         ticktextside=None,
#         bgcolor=None,
#         bordercolor=None,
#         borderwidth=None,
#         dtick=None,
#         exponentformat=None,
#         len=None,
#         lenmode=None,
#         nticks=None,
#         outlinecolor=None,
#         outlinewidth=None,
#         separatethousands=None,
#         showexponent=None,
#         showticklabels=None,
#         showtickprefix=None,
#         showticksuffix=None,
#         thickness=None,
#         thicknessmode=None,
#         tick0=None,
#         tickangle=None,
#         tickcolor=None,
#         tickfont=None,
#         tickformat=None,
#         tickformatstops=None,
#         tickformatstopdefaults=None,
#         ticklen=None,
#         tickmode=None,
#         tickprefix=None,
#         ticks=None,
#         ticksuffix=None,
#         ticktext=None,
#         ticktextsrc=None,
#         tickvals=None,
#         tickvalssrc=None,
#         tickwidth=None,
#         title=None,
#         titlefont=None,
#         titleside=None,
#         x=None,
#         xanchor=None,
#         xpad=None,
#         y=None,
#         yanchor=None,
#         ypad=None,
#         **kwargs
#     ):
#     ColorBar=go.heatmap.ColorBar
#     super(ColorBar, self).__init__("colorbar")

#     # Validate arg
#     # ------------
#     if arg is None:
#         arg = {}
#     elif isinstance(arg, self.__class__):
#         arg = arg.to_plotly_study_json()
#     elif isinstance(arg, dict):
#         arg = _copy.copy(arg)
#     else:
#         raise ValueError(
#             """\
# The first argument to the plotly_study.graph_objs.heatmap.ColorBar 
# constructor must be a dict or 
# an instance of plotly_study.graph_objs.heatmap.ColorBar"""
#         )

#     # Handle skip_invalid
#     # -------------------
#     self._skip_invalid = kwargs.pop("skip_invalid", False)

#     # Import validators
#     # -----------------
#     from plotly_study.validators.heatmap import colorbar as v_colorbar

#     # Initialize validators
#     # ---------------------
#     self._validators["bgcolor"] = v_colorbar.BgcolorValidator()
#     self._validators["bordercolor"] = v_colorbar.BordercolorValidator()
#     self._validators["borderwidth"] = v_colorbar.BorderwidthValidator()
#     self._validators["dtick"] = v_colorbar.DtickValidator()
#     self._validators["exponentformat"] = v_colorbar.ExponentformatValidator()
#     self._validators["len"] = v_colorbar.LenValidator()
#     self._validators["lenmode"] = v_colorbar.LenmodeValidator()
#     self._validators["nticks"] = v_colorbar.NticksValidator()
#     self._validators["outlinecolor"] = v_colorbar.OutlinecolorValidator()
#     self._validators["outlinewidth"] = v_colorbar.OutlinewidthValidator()
#     self._validators["separatethousands"] = v_colorbar.SeparatethousandsValidator()
#     self._validators["showexponent"] = v_colorbar.ShowexponentValidator()
#     self._validators["showticklabels"] = v_colorbar.ShowticklabelsValidator()
#     self._validators["showtickprefix"] = v_colorbar.ShowtickprefixValidator()
#     self._validators["showticksuffix"] = v_colorbar.ShowticksuffixValidator()
#     self._validators["thickness"] = v_colorbar.ThicknessValidator()
#     self._validators["thicknessmode"] = v_colorbar.ThicknessmodeValidator()
#     self._validators["tick0"] = v_colorbar.Tick0Validator()
#     self._validators["tickangle"] = v_colorbar.TickangleValidator()
#     self._validators["tickcolor"] = v_colorbar.TickcolorValidator()
#     self._validators["tickfont"] = v_colorbar.TickfontValidator()
#     self._validators["tickformat"] = v_colorbar.TickformatValidator()
#     self._validators["tickformatstops"] = v_colorbar.TickformatstopsValidator()
#     self._validators[
#         "tickformatstopdefaults"
#     ] = v_colorbar.TickformatstopValidator()
#     self._validators["ticklen"] = v_colorbar.TicklenValidator()
#     self._validators["tickmode"] = v_colorbar.TickmodeValidator()
#     self._validators["tickprefix"] = v_colorbar.TickprefixValidator()
#     self._validators["ticks"] = v_colorbar.TicksValidator()
#     self._validators["ticksuffix"] = v_colorbar.TicksuffixValidator()
#     self._validators["ticktext"] = v_colorbar.TicktextValidator()
#     self._validators["ticktextsrc"] = v_colorbar.TicktextsrcValidator()
#     self._validators["tickvals"] = v_colorbar.TickvalsValidator()
#     self._validators["tickvalssrc"] = v_colorbar.TickvalssrcValidator()
#     self._validators["tickwidth"] = v_colorbar.TickwidthValidator()
#     self._validators["title"] = v_colorbar.TitleValidator()
#     self._validators["x"] = v_colorbar.XValidator()
#     self._validators["xanchor"] = v_colorbar.XanchorValidator()
#     self._validators["xpad"] = v_colorbar.XpadValidator()
#     self._validators["y"] = v_colorbar.YValidator()
#     self._validators["yanchor"] = v_colorbar.YanchorValidator()
#     self._validators["ypad"] = v_colorbar.YpadValidator()
#     self._validators["ticktextside"]=v_colorbar.TickTextSideValidator()
#     # Populate data dict with properties
#     # ----------------------------------
#     _v = arg.pop("bgcolor", None)
#     self["bgcolor"] = bgcolor if bgcolor is not None else _v
#     _v = arg.pop("bordercolor", None)
#     self["bordercolor"] = bordercolor if bordercolor is not None else _v
#     _v = arg.pop("borderwidth", None)
#     self["borderwidth"] = borderwidth if borderwidth is not None else _v
#     _v = arg.pop("dtick", None)
#     self["dtick"] = dtick if dtick is not None else _v
#     _v = arg.pop("exponentformat", None)
#     self["exponentformat"] = exponentformat if exponentformat is not None else _v
#     _v = arg.pop("len", None)
#     self["len"] = len if len is not None else _v
#     _v = arg.pop("lenmode", None)
#     self["lenmode"] = lenmode if lenmode is not None else _v
#     _v = arg.pop("nticks", None)
#     self["nticks"] = nticks if nticks is not None else _v
#     _v = arg.pop("outlinecolor", None)
#     self["outlinecolor"] = outlinecolor if outlinecolor is not None else _v
#     _v = arg.pop("outlinewidth", None)
#     self["outlinewidth"] = outlinewidth if outlinewidth is not None else _v
#     _v = arg.pop("separatethousands", None)
#     self["separatethousands"] = (
#         separatethousands if separatethousands is not None else _v
#     )
#     _v = arg.pop("showexponent", None)
#     self["showexponent"] = showexponent if showexponent is not None else _v
#     _v = arg.pop("showticklabels", None)
#     self["showticklabels"] = showticklabels if showticklabels is not None else _v
#     _v = arg.pop("showtickprefix", None)
#     self["showtickprefix"] = showtickprefix if showtickprefix is not None else _v
#     _v = arg.pop("showticksuffix", None)
#     self["showticksuffix"] = showticksuffix if showticksuffix is not None else _v
#     _v = arg.pop("thickness", None)
#     self["thickness"] = thickness if thickness is not None else _v
#     _v = arg.pop("thicknessmode", None)
#     self["thicknessmode"] = thicknessmode if thicknessmode is not None else _v
#     _v = arg.pop("tick0", None)
#     self["tick0"] = tick0 if tick0 is not None else _v
#     _v = arg.pop("tickangle", None)
#     self["tickangle"] = tickangle if tickangle is not None else _v
#     _v = arg.pop("tickcolor", None)
#     self["tickcolor"] = tickcolor if tickcolor is not None else _v
#     _v = arg.pop("tickfont", None)
#     self["tickfont"] = tickfont if tickfont is not None else _v
#     _v = arg.pop("tickformat", None)
#     self["tickformat"] = tickformat if tickformat is not None else _v
#     _v = arg.pop("tickformatstops", None)
#     self["tickformatstops"] = tickformatstops if tickformatstops is not None else _v
#     _v = arg.pop("tickformatstopdefaults", None)
#     self["tickformatstopdefaults"] = (
#         tickformatstopdefaults if tickformatstopdefaults is not None else _v
#     )
#     _v = arg.pop("ticklen", None)
#     self["ticklen"] = ticklen if ticklen is not None else _v
#     _v = arg.pop("tickmode", None)
#     self["tickmode"] = tickmode if tickmode is not None else _v
#     _v = arg.pop("tickprefix", None)
#     self["tickprefix"] = tickprefix if tickprefix is not None else _v
#     _v = arg.pop("ticks", None)
#     self["ticks"] = ticks if ticks is not None else _v
#     _v = arg.pop("ticksuffix", None)
#     self["ticksuffix"] = ticksuffix if ticksuffix is not None else _v
#     _v = arg.pop("ticktext", None)
#     self["ticktext"] = ticktext if ticktext is not None else _v
#     _v = arg.pop("ticktextsrc", None)
#     self["ticktextsrc"] = ticktextsrc if ticktextsrc is not None else _v
#     _v = arg.pop("tickvals", None)
#     self["tickvals"] = tickvals if tickvals is not None else _v
#     _v = arg.pop("tickvalssrc", None)
#     self["tickvalssrc"] = tickvalssrc if tickvalssrc is not None else _v
#     _v = arg.pop("tickwidth", None)
#     self["tickwidth"] = tickwidth if tickwidth is not None else _v
#     _v = arg.pop("title", None)
#     self["title"] = title if title is not None else _v
#     _v = arg.pop("titlefont", None)
#     _v = titlefont if titlefont is not None else _v
#     if _v is not None:
#         self["titlefont"] = _v
#     _v = arg.pop("titleside", None)
#     _v = titleside if titleside is not None else _v
#     if _v is not None:
#         self["titleside"] = _v
#     _v = arg.pop("x", None)
#     self["x"] = x if x is not None else _v
#     _v = arg.pop("xanchor", None)
#     self["xanchor"] = xanchor if xanchor is not None else _v
#     _v = arg.pop("xpad", None)
#     self["xpad"] = xpad if xpad is not None else _v
#     _v = arg.pop("y", None)
#     self["y"] = y if y is not None else _v
#     _v = arg.pop("yanchor", None)
#     self["yanchor"] = yanchor if yanchor is not None else _v
#     _v = arg.pop("ypad", None)
#     self["ypad"] = ypad if ypad is not None else _v
#     _v = arg.pop("ticktextside", None)
#     self["ticktextside"] = ticktextside if ticktextside is not None else _v
#     # Process unknown kwargs
#     # ----------------------
#     self._process_kwargs(**dict(arg, **kwargs))

#     # Reset skip_invalid
#     # ------------------
#     self._skip_invalid = False
# uu=property(lambda self: self["ticktextside"])
# uu=uu.setter(lambda self,val:setattr(self,"ticktextside",val))
# go.heatmap.ColorBar.ticktextside=uu
# go.heatmap.ColorBar.__init__=__init3__
# import _plotly_study_utils.basevalidators
# import plotly_study
# class TickTextSideValidator(_plotly_study_utils.basevalidators.EnumeratedValidator):
#     def __init__(
#         self, plotly_study_name="ticktextside", parent_name="heatmap.colorbar", **kwargs
#     ):
#         super(TickTextSideValidator, self).__init__(
#             plotly_study_name=plotly_study_name,
#             parent_name=parent_name,
#             edit_type=kwargs.pop("edit_type", "colorbars"),
#             implied_edits=kwargs.pop("implied_edits", {}),
#             role=kwargs.pop("role", "info"),
#             values=kwargs.pop("values", ["right", "left"]),
#             **kwargs
# #         )
# from plotly_study.validators.heatmap import colorbar as vCol
# vCol.TickTextSideValidator= TickTextSideValidator
# def __init2__(
#         self,
#         arg=None,
#         ticktextside=None,
#         bgcolor=None,
#         bordercolor=None,
#         borderwidth=None,
#         dtick=None,
#         exponentformat=None,
#         len=None,
#         lenmode=None,
#         nticks=None,
#         outlinecolor=None,
#         outlinewidth=None,
#         separatethousands=None,
#         showexponent=None,
#         showticklabels=None,
#         showtickprefix=None,
#         showticksuffix=None,
#         thickness=None,
#         thicknessmode=None,
#         tick0=None,
#         tickangle=None,
#         tickcolor=None,
#         tickfont=None,
#         tickformat=None,
#         tickformatstops=None,
#         tickformatstopdefaults=None,
#         ticklen=None,
#         tickmode=None,
#         tickprefix=None,
#         ticks=None,
#         ticksuffix=None,
#         ticktext=None,
#         ticktextsrc=None,
#         tickvals=None,
#         tickvalssrc=None,
#         tickwidth=None,
#         title=None,
#         titlefont=None,
#         titleside=None,
#         x=None,
#         xanchor=None,
#         xpad=None,
#         y=None,
#         yanchor=None,
#         ypad=None,
#         **kwargs):
#         ColorBar=go.scatter.marker.ColorBar
#         super(ColorBar, self).__init__("colorbar")

#         # Validate arg
#         # ------------
#         if arg is None:
#             arg = {}
#         elif isinstance(arg, self.__class__):
#             arg = arg.to_plotly_study_json()
#         elif isinstance(arg, dict):
#             arg = _copy.copy(arg)
#         else:
#             raise ValueError(
#                 """\
# The first argument to the plotly_study.graph_objs.scatter.marker.ColorBar 
# constructor must be a dict or 
# an instance of plotly_study.graph_objs.scatter.marker.ColorBar"""
#             )

#         # Handle skip_invalid
#         # -------------------
#         self._skip_invalid = kwargs.pop("skip_invalid", False)

#         # Import validators
#         # -----------------
#         from plotly_study.validators.scatter.marker import colorbar as v_colorbar

#         # Initialize validators
#         # ---------------------
#         self._validators["bgcolor"] = v_colorbar.BgcolorValidator()
#         self._validators["bordercolor"] = v_colorbar.BordercolorValidator()
#         self._validators["borderwidth"] = v_colorbar.BorderwidthValidator()
#         self._validators["dtick"] = v_colorbar.DtickValidator()
#         self._validators["exponentformat"] = v_colorbar.ExponentformatValidator()
#         self._validators["len"] = v_colorbar.LenValidator()
#         self._validators["lenmode"] = v_colorbar.LenmodeValidator()
#         self._validators["nticks"] = v_colorbar.NticksValidator()
#         self._validators["outlinecolor"] = v_colorbar.OutlinecolorValidator()
#         self._validators["outlinewidth"] = v_colorbar.OutlinewidthValidator()
#         self._validators["separatethousands"] = v_colorbar.SeparatethousandsValidator()
#         self._validators["showexponent"] = v_colorbar.ShowexponentValidator()
#         self._validators["showticklabels"] = v_colorbar.ShowticklabelsValidator()
#         self._validators["showtickprefix"] = v_colorbar.ShowtickprefixValidator()
#         self._validators["showticksuffix"] = v_colorbar.ShowticksuffixValidator()
#         self._validators["thickness"] = v_colorbar.ThicknessValidator()
#         self._validators["thicknessmode"] = v_colorbar.ThicknessmodeValidator()
#         self._validators["tick0"] = v_colorbar.Tick0Validator()
#         self._validators["tickangle"] = v_colorbar.TickangleValidator()
#         self._validators["tickcolor"] = v_colorbar.TickcolorValidator()
#         self._validators["tickfont"] = v_colorbar.TickfontValidator()
#         self._validators["tickformat"] = v_colorbar.TickformatValidator()
#         self._validators["tickformatstops"] = v_colorbar.TickformatstopsValidator()
#         self._validators[
#             "tickformatstopdefaults"
#         ] = v_colorbar.TickformatstopValidator()
#         self._validators["ticklen"] = v_colorbar.TicklenValidator()
#         self._validators["tickmode"] = v_colorbar.TickmodeValidator()
#         self._validators["tickprefix"] = v_colorbar.TickprefixValidator()
#         self._validators["ticks"] = v_colorbar.TicksValidator()
#         self._validators["ticksuffix"] = v_colorbar.TicksuffixValidator()
#         self._validators["ticktext"] = v_colorbar.TicktextValidator()
#         self._validators["ticktextsrc"] = v_colorbar.TicktextsrcValidator()
#         self._validators["tickvals"] = v_colorbar.TickvalsValidator()
#         self._validators["tickvalssrc"] = v_colorbar.TickvalssrcValidator()
#         self._validators["tickwidth"] = v_colorbar.TickwidthValidator()
#         self._validators["title"] = v_colorbar.TitleValidator()
#         self._validators["x"] = v_colorbar.XValidator()
#         self._validators["xanchor"] = v_colorbar.XanchorValidator()
#         self._validators["xpad"] = v_colorbar.XpadValidator()
#         self._validators["y"] = v_colorbar.YValidator()
#         self._validators["yanchor"] = v_colorbar.YanchorValidator()
#         self._validators["ypad"] = v_colorbar.YpadValidator()
#         self._validators["ticktextside"]=v_colorbar.TickTextSideValidator()
#         # Populate data dict with properties
#         # ----------------------------------
#         _v = arg.pop("bgcolor", None)
#         self["bgcolor"] = bgcolor if bgcolor is not None else _v
#         _v = arg.pop("bordercolor", None)
#         self["bordercolor"] = bordercolor if bordercolor is not None else _v
#         _v = arg.pop("borderwidth", None)
#         self["borderwidth"] = borderwidth if borderwidth is not None else _v
#         _v = arg.pop("dtick", None)
#         self["dtick"] = dtick if dtick is not None else _v
#         _v = arg.pop("exponentformat", None)
#         self["exponentformat"] = exponentformat if exponentformat is not None else _v
#         _v = arg.pop("len", None)
#         self["len"] = len if len is not None else _v
#         _v = arg.pop("lenmode", None)
#         self["lenmode"] = lenmode if lenmode is not None else _v
#         _v = arg.pop("nticks", None)
#         self["nticks"] = nticks if nticks is not None else _v
#         _v = arg.pop("outlinecolor", None)
#         self["outlinecolor"] = outlinecolor if outlinecolor is not None else _v
#         _v = arg.pop("outlinewidth", None)
#         self["outlinewidth"] = outlinewidth if outlinewidth is not None else _v
#         _v = arg.pop("separatethousands", None)
#         self["separatethousands"] = (
#             separatethousands if separatethousands is not None else _v
#         )
#         _v = arg.pop("showexponent", None)
#         self["showexponent"] = showexponent if showexponent is not None else _v
#         _v = arg.pop("showticklabels", None)
#         self["showticklabels"] = showticklabels if showticklabels is not None else _v
#         _v = arg.pop("showtickprefix", None)
#         self["showtickprefix"] = showtickprefix if showtickprefix is not None else _v
#         _v = arg.pop("showticksuffix", None)
#         self["showticksuffix"] = showticksuffix if showticksuffix is not None else _v
#         _v = arg.pop("thickness", None)
#         self["thickness"] = thickness if thickness is not None else _v
#         _v = arg.pop("thicknessmode", None)
#         self["thicknessmode"] = thicknessmode if thicknessmode is not None else _v
#         _v = arg.pop("tick0", None)
#         self["tick0"] = tick0 if tick0 is not None else _v
#         _v = arg.pop("tickangle", None)
#         self["tickangle"] = tickangle if tickangle is not None else _v
#         _v = arg.pop("tickcolor", None)
#         self["tickcolor"] = tickcolor if tickcolor is not None else _v
#         _v = arg.pop("tickfont", None)
#         self["tickfont"] = tickfont if tickfont is not None else _v
#         _v = arg.pop("tickformat", None)
#         self["tickformat"] = tickformat if tickformat is not None else _v
#         _v = arg.pop("tickformatstops", None)
#         self["tickformatstops"] = tickformatstops if tickformatstops is not None else _v
#         _v = arg.pop("tickformatstopdefaults", None)
#         self["tickformatstopdefaults"] = (
#             tickformatstopdefaults if tickformatstopdefaults is not None else _v
#         )
#         _v = arg.pop("ticklen", None)
#         self["ticklen"] = ticklen if ticklen is not None else _v
#         _v = arg.pop("tickmode", None)
#         self["tickmode"] = tickmode if tickmode is not None else _v
#         _v = arg.pop("tickprefix", None)
#         self["tickprefix"] = tickprefix if tickprefix is not None else _v
#         _v = arg.pop("ticks", None)
#         self["ticks"] = ticks if ticks is not None else _v
#         _v = arg.pop("ticksuffix", None)
#         self["ticksuffix"] = ticksuffix if ticksuffix is not None else _v
#         _v = arg.pop("ticktext", None)
#         self["ticktext"] = ticktext if ticktext is not None else _v
#         _v = arg.pop("ticktextsrc", None)
#         self["ticktextsrc"] = ticktextsrc if ticktextsrc is not None else _v
#         _v = arg.pop("tickvals", None)
#         self["tickvals"] = tickvals if tickvals is not None else _v
#         _v = arg.pop("tickvalssrc", None)
#         self["tickvalssrc"] = tickvalssrc if tickvalssrc is not None else _v
#         _v = arg.pop("tickwidth", None)
#         self["tickwidth"] = tickwidth if tickwidth is not None else _v
#         _v = arg.pop("title", None)
#         self["title"] = title if title is not None else _v
#         _v = arg.pop("titlefont", None)
#         _v = titlefont if titlefont is not None else _v
#         if _v is not None:
#             self["titlefont"] = _v
#         _v = arg.pop("titleside", None)
#         _v = titleside if titleside is not None else _v
#         if _v is not None:
#             self["titleside"] = _v
#         _v = arg.pop("x", None)
#         self["x"] = x if x is not None else _v
#         _v = arg.pop("xanchor", None)
#         self["xanchor"] = xanchor if xanchor is not None else _v
#         _v = arg.pop("xpad", None)
#         self["xpad"] = xpad if xpad is not None else _v
#         _v = arg.pop("y", None)
#         self["y"] = y if y is not None else _v
#         _v = arg.pop("yanchor", None)
#         self["yanchor"] = yanchor if yanchor is not None else _v
#         _v = arg.pop("ypad", None)
#         self["ypad"] = ypad if ypad is not None else _v
#         _v = arg.pop("ticktextside", None)
#         self["ticktextside"] = ticktextside if ticktextside is not None else _v

#         # Process unknown kwargs
#         # ----------------------
#         self._process_kwargs(**dict(arg, **kwargs))

#         # Reset skip_invalid
#         # ------------------
#         self._skip_invalid = False
# uu=property(lambda self: self["ticktextside"])
# uu=uu.setter(lambda self,val:setattr(self,"ticktextside",val))
# go.scatter.marker.ColorBar.ticktextside=uu
# go.scatter.marker.ColorBar.__init__=__init2__
# # import _plotly_study_utils.basevalidators
# # import plotly_study
# # class TickTextSideValidator2(_plotly_study_utils.basevalidators.EnumeratedValidator):
# #     def __init__(
# #         self, plotly_study_name="ticktextside", parent_name="scatter.marker.colorbar", **kwargs
# #     ):
# #         super(TickTextSideValidator2, self).__init__(
# #             plotly_study_name=plotly_study_name,
# #             parent_name=parent_name,
# #             edit_type=kwargs.pop("edit_type", "colorbars"),
# #             implied_edits=kwargs.pop("implied_edits", {}),
# #             role=kwargs.pop("role", "info"),
# #             values=kwargs.pop("values", ["right", "left"]),
# #             **kwargs
# #         )
# # from plotly_study.validators.scatter.marker import colorbar as vCol
# # vCol.TickTextSideValidator= TickTextSideValidator2
# go.scatter.marker.ColorBar.OLD__INIT__=go.scatter.marker.ColorBar.__init__
# function ttt(l, tab=""){
#   var srti=tab
#   var val = null;
#     for (const [fruit, count] of  Object.entries(l)) {
#       if(Array.isArray(count)){
#           val="[]"
#       }else if(count == true && typeof(count) === "boolean"){
#           val="True"
#       }else if(count==false && typeof(count) === "boolean"){
#           val="False"
#       }else if(typeof(count)=="string" || typeof(count)=="number"){
#       val = count
#       }
#       else if (typeof(count) === 'object' && count !== null){
#           if (Object.keys(count).length==0){
#               val="dict()"
#           }else{
#               val="dict(\n"+ttt(count,tab+"\t")+"\n)"
#           }
#       }
#       //console.log(fruit+" "+typeof(count)+" "+val)
#       srti=srti+fruit+"="+val+"\n"+tab
#     }
#     return srti 
# }    