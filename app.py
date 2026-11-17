"""CNT735 | Assuntos 11–16 | Streamlit."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from cnt735.modelos import ajustar, prever, matriz, anova_ortogonal, extremos

st.set_page_config(page_title='CNT735 • Laboratório DoE',layout='wide',page_icon='🧪')
ROOT=Path(__file__).parent
BB=[22,24,4,6,16,18,10,12,8,26,2,20,14]
# Escala cromática consistente entre os gráficos quantitativos de superfície.
# A mesma paleta e os mesmos limites são compartilhados por 3D, contorno e calor.
ESCALA_ARCO_IRIS = 'Rainbow'
TITULOS={'11':'11 · Modelo fatorial 2²','12':'12 · Réplicas e superfície de resposta','13':'13 · Planejamento fatorial 3³','14':'14 · Fatorial misto 3×2×3 (CdS)','15':'15 · Box–Behnken × 3³','16':'16 · Blocos e confundimento'}
@st.cache_data
def original(k):return pd.DataFrame(json.loads((ROOT/'data'/f'assunto{k}.json').read_text()))

def editar(k,origem=None):
 d=(origem if origem is not None else original(k)).copy()
 st.info('Edite somente as respostas **Y**. Os fatores permanecem codificados conforme o planejamento. Alterações recalculam todas as análises.')
 show=st.data_editor(d,key='editor_'+k,hide_index=True,use_container_width=True,num_rows='fixed',disabled=[c for c in d.columns if c!='Y'],column_config={'Y':st.column_config.NumberColumn('Y — resposta (editável)',format='%.5f',required=True)})
 if st.button('Restaurar dados originais',key='reset'+k):
  del st.session_state['editor_'+k];st.rerun()
 if show['Y'].isna().any():st.warning('Preencha todas as respostas Y.');st.stop()
 return show

def tabela(nome,d):
 st.markdown('**'+nome+'**');st.dataframe(d,use_container_width=True)

def metricas(m):
 a,b,c,d,e=st.columns(5)
 a.metric('N',m['n']);b.metric('Parâmetros',m['p']);c.metric('gl residuais',m['gl']);d.metric('R²',f"{m['r2']:.5f}");e.metric('RMSE treino',f"{m['rmse']:.4f}")
 st.caption('R² ajustado: '+(f"{m['r2aj']:.5f}" if np.isfinite(m['r2aj']) else 'não definido')+' | RSE (√SSE/gl): '+(f"{m['rse']:.4f}" if np.isfinite(m['rse']) else 'não estimável'))

def grafico_coef(m,com_ep=True):
 c=m['coef'].query('Termo != "1"').copy();kw={'error_y':'EP'} if com_ep and c.EP.notna().all() else {}
 st.plotly_chart(px.bar(c,x='Termo',y='Coeficiente',title='Coeficientes de regressão',**kw),use_container_width=True)

def diagnosticos(m,ordem=True):
 df=m['pred'];y=df['Y'];yh=df['Predito'];res=df['Residuo']
 l,r=st.columns(2)
 with l:
  fig=px.scatter(df,x='Y',y='Predito',title='Observado × predito');lo=min(y.min(),yh.min());hi=max(y.max(),yh.max());fig.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode='lines',name='Identidade'));st.plotly_chart(fig,use_container_width=True)
  st.plotly_chart(px.histogram(df,x='Residuo',nbins=min(12,max(4,len(df)//2)),title='Distribuição dos resíduos'),use_container_width=True)
 with r:
  fig=px.scatter(df,x='Predito',y='Residuo',title='Resíduos × valores preditos');fig.add_hline(y=0,line_dash='dash');st.plotly_chart(fig,use_container_width=True)
  if len(res)>=3 and m['gl']>0:
   q=stats.norm.ppf((np.arange(1,len(res)+1)-.5)/len(res));qq=pd.DataFrame({'Quantil normal teórico':q,'Resíduo ordenado':np.sort(res)})
   st.plotly_chart(px.scatter(qq,x='Quantil normal teórico',y='Resíduo ordenado',title='Q-Q plot'),use_container_width=True)
  else:st.info('Q-Q sem interpretação inferencial: modelo saturado ou amostra insuficiente.')
 if ordem:
  d=df.copy();d['Ordem']=np.arange(1,len(df)+1)
  st.plotly_chart(px.scatter(d,x='Ordem',y='Residuo',title='Resíduos por ordem da planilha (não equivale necessariamente à ordem de execução)'),use_container_width=True)
 tabela('Predições e resíduos',df)

def passos(m):
 with st.expander('Álgebra matricial — todas as etapas',expanded=False):
  st.latex(r'\hat\beta=(X^TX)^{-1}X^Ty')
  for nome,val in [('1 · X',m['X']),('2 · Xᵀ',m['Xt']),('3 · XᵀX',m['XtX']),('4 · (XᵀX)⁻¹',m['XtX_inv']),('5 · Xᵀy',m['Xty'].to_frame()),('6 · Coeficientes',m['coef'])]:tabela(nome,val)

def fatores(df,nomes):
 cols=st.columns(len(nomes))
 for col,nome in zip(cols,nomes):
  means=df.groupby(nome,as_index=False)['Y'].mean()
  with col:st.plotly_chart(px.line(means,x=nome,y='Y',markers=True,title=f'Efeito principal — {nome}'),use_container_width=True)

def interacoes(df,pares):
 for a,b in pares:
  z=df.groupby([a,b],as_index=False)['Y'].mean();z[b]=z[b].astype(str)
  st.plotly_chart(px.line(z,x=a,y='Y',color=b,markers=True,title=f'Interação observada {a}×{b} (médias marginais)'),use_container_width=True)

def superficie(m,x,y,fixos=None,delta=None,passo=41):
 fixos=fixos or {}
 axis=np.linspace(-1,1,passo)
 gx,gy=np.meshgrid(axis,axis)
 d=pd.DataFrame({x:gx.ravel(),y:gy.ravel()})
 for k,v in fixos.items():d[k]=v
 z=(prever(m,d) if delta is None else prever(m,d)-prever(delta,d)).reshape(gx.shape)
 rotulo='ΔY' if delta is not None else 'Y previsto'
 minimo,maximo=float(np.min(z)),float(np.max(z))
 if delta is not None:
  # A escala permanece centrada em zero para a diferença entre modelos.
  limite=max(abs(minimo),abs(maximo),1e-12)
  cmin,cmax=-limite,limite
 else:
  # Limites comuns: a mesma cor representa o mesmo valor nos três gráficos.
  cmin,cmax=minimo,maximo
  if np.isclose(cmin,cmax):
   margem=max(abs(cmin)*0.01,0.01)
   cmin-=margem;cmax+=margem
 a,b=st.columns(2)
 with a:
  fig=go.Figure(go.Surface(x=gx,y=gy,z=z,colorscale=ESCALA_ARCO_IRIS,cmin=cmin,cmax=cmax,
      colorbar=dict(title=rotulo)))
  fig.update_layout(title=f'Superfície 3D — {x} × {y}',scene=dict(xaxis_title=x,yaxis_title=y,zaxis_title=rotulo),height=470)
  st.plotly_chart(fig,use_container_width=True)
 with b:
  fig=go.Figure(go.Contour(x=axis,y=axis,z=z,colorscale=ESCALA_ARCO_IRIS,zmin=cmin,zmax=cmax,
      colorbar=dict(title=rotulo),contours=dict(showlabels=True)))
  fig.update_layout(title='Mapa de contorno',xaxis_title=x,yaxis_title=y,height=470)
  st.plotly_chart(fig,use_container_width=True)
 fig=go.Figure(go.Heatmap(x=axis,y=axis,z=z,colorscale=ESCALA_ARCO_IRIS,zmin=cmin,zmax=cmax,
     colorbar=dict(title=rotulo),hovertemplate=f'{x}=%{{x:.3f}}<br>{y}=%{{y:.3f}}<br>{rotulo}=%{{z:.4f}}<extra></extra>'))
 fig.update_layout(title='Mapa de calor — escala arco-íris',xaxis_title=x,yaxis_title=y,height=470)
 st.plotly_chart(fig,use_container_width=True)
 if delta is not None:
  st.caption('Escala arco-íris simétrica em torno de ΔY = 0: permite comparar diferenças positivas e negativas entre os modelos.')
 st.caption(f'Malha determinística {passo}×{passo}. Máximo previsto na fatia: {z.max():.4f} em {x}={axis[np.unravel_index(z.argmax(),z.shape)[1]]:.3f}; {y}={axis[np.unravel_index(z.argmax(),z.shape)[0]]:.3f}. Mínimo previsto: {z.min():.4f}. Ótimo numérico ≠ ótimo validado.')
 return z

def baixas(df,m,nome):
 data=m['pred'].to_csv(index=False).encode('utf-8-sig');st.download_button('⬇ Baixar tabela e predições (CSV)',data,file_name=f'CNT735_{nome}.csv',mime='text/csv')

def do11():
 df=editar('11');m=ajustar(df,'11');metricas(m)
 a,b,c,d=st.tabs(['Matrizes','Efeitos','Predições e gráficos','Interpretação'])
 with a:passos(m);tabela('Coeficientes',m['coef'])
 with b:
  z=m['coef'].query('Termo!="1"')[['Termo','Coeficiente']].copy();z['Efeito']=2*z['Coeficiente'];tabela('Efeito = 2 × coeficiente',z);grafico_coef(m)
 with c:diagnosticos(m)
 with d:st.warning('Modelo saturado: N=p=4. Os resíduos são zero por construção, mas não há graus de liberdade para estimar erro experimental, ICs ou capacidade preditiva fora das condições observadas. Necessárias réplicas independentes e confirmação externa.')
 baixas(df,m,'assunto11')

def do12():
 st.warning('Fonte: planilha Assunto 12. Réplicas 2 e 3 foram calculadas por +5% e −3% dos valores-base; **não são réplicas experimentais independentes**. ICs e erros padrão são apenas demonstrações aritméticas.')
 df=editar('12');groups=[]
 for (c,t),g in df.groupby(['C','T']):
  y=g.Y.to_numpy();n=len(y);s=y.std(ddof=1) if n>1 else np.nan;ep=s/np.sqrt(n);h=stats.t.ppf(.975,n-1)*ep if n>1 else np.nan
  groups.append({'C':c,'T':t,'N':n,'Media':y.mean(),'DP':s,'EP':ep,'IC95_inf':y.mean()-h,'IC95_sup':y.mean()+h,'Margem95':h})
 gg=pd.DataFrame(groups);tabela('Resumo por condição',gg)
 gg['Condicao']=gg['C'].astype(str)+' / '+gg['T'].astype(str)
 st.plotly_chart(px.scatter(gg,x='Condicao',y='Media',error_y='Margem95',title='Médias e IC95% formais (réplicas determinísticas)'),use_container_width=True)
 m1=ajustar(df,'12_linear');m2=ajustar(df,'12_curva')
 st.info('Na matriz original, C²=T² em todos os pontos. O quadrático completo de seis parâmetros NÃO tem solução única. Ajustamos a curvatura identificável Q=(C²+T²)/2, com cinco coeficientes; seu coeficiente representa b₁₁+b₂₂.')
 resumo=pd.DataFrame([{'Modelo':'Primeira ordem + CT','R²':m1['r2'],'R² ajustado':m1['r2aj'],'RMSE':m1['rmse'],'p':m1['p']},{'Modelo':'Curvatura conjunta','R²':m2['r2'],'R² ajustado':m2['r2aj'],'RMSE':m2['rmse'],'p':m2['p']}]);tabela('Comparação de modelos',resumo)
 tabs=st.tabs(['Primeira ordem','Curvatura','Superfície','Discussão'])
 for tab,m in zip(tabs[:2],[m1,m2]):
  with tab:metricas(m);passos(m);grafico_coef(m);diagnosticos(m)
 with tabs[2]:
  qual=st.selectbox('Modelo a representar',['Primeira ordem','Curvatura']);superficie(m1 if qual=='Primeira ordem' else m2,'C','T')
 with tabs[3]:st.warning('O ganho aparente de R² deve ser confrontado com complexidade, validação independente e replicação autêntica. Superfícies representam extensão modelada; não há dados fora dos vértices e do centro.')
 baixas(df,m2,'assunto12')

def do13():
 st.info('As 27 respostas foram recuperadas da planilha do Assunto 15, que inclui a base histórica 3³. A planilha anexada do Assunto 13 tinha a coluna de respostas vazia. A/S/P são codificações derivadas dos níveis físicos.')
 df=editar('13');m=ajustar(df,'13');metricas(m)
 tabs=st.tabs(['Modelo e matrizes','Diagnósticos','Efeitos e interações','Superfícies e otimização'])
 with tabs[0]:passos(m);tabela('Coeficientes e IC95%',m['coef']);grafico_coef(m)
 with tabs[1]:diagnosticos(m)
 with tabs[2]:fatores(df,['A','S','P']);interacoes(df,[('A','S'),('A','P'),('S','P')])
 with tabs[3]:
  a,b=st.selectbox('Fatores na superfície',[('A','S'),('A','P'),('S','P')],format_func=lambda t:' × '.join(t));fix=({'A','S','P'}-{a,b}).pop();nivel=st.select_slider(f'{fix} fixo',[-1.,0.,1.],value=0.);superficie(m,a,b,{fix:nivel})
  grid=extremos(m,['A','S','P'],resolucao=17);tabela('Busca aproximada: maior e menor resposta na malha',pd.concat([grid.nlargest(1,'Predito'),grid.nsmallest(1,'Predito')]))
  st.caption('Região robusta exige análise de sensibilidade e confirmação. Interações e efeitos são associativos no contexto do desenho experimental.')
 baixas(df,m,'assunto13')

def do14():
 st.warning('Os 18 valores iniciais de eficiência de CdS são **SIMULADOS** conforme a planilha da Aula 17. Substitua todas as respostas por dados reais antes de realizar conclusões experimentais. CdS contém cádmio: atenção à segurança.')
 df=editar('14');m=ajustar(df,'14');metricas(m)
 tabs=st.tabs(['Modelo','Diagnósticos','Efeitos','Superfícies por concentração'])
 with tabs[0]:passos(m);tabela('Coeficientes e IC95%',m['coef']);grafico_coef(m);st.info('C² foi excluído: C assume apenas ±1, logo C²=1 (confundido com intercepto).')
 with tabs[1]:diagnosticos(m)
 with tabs[2]:fatores(df,['D','C','E']);interacoes(df,[('D','C'),('D','E'),('E','C')])
 with tabs[3]:
  for c in [-1.,1.]:
   st.subheader(f'Concentração C={c:+g}');superficie(m,'D','E',{'C':c})
  st.latex(r'\Delta Y(D,E)=Y(C=+1)-Y(C=-1)=2(b_C+b_{DC}D+b_{CE}E)')
  st.caption('Como C interage com D e E, a diferença das superfícies pode variar espacialmente; os dados devem sustentar a conclusão.')
 baixas(df,m,'assunto14')

def do15():
 st.info('Base compartilhada com Assunto 13: 27 observações e subconjunto Box–Behnken com IDs originais preservados. Se alterar Y aqui, a edição vale nesta atividade; a base de 13 não é sobrescrita.')
 df=editar('15',original('13'));sub=df.set_index('ID').loc[BB].reset_index();m27=ajustar(df,'13');m13=ajustar(sub,'13')
 resumo=pd.DataFrame([{'Plano':'3³ completo','N':m27['n'],'Parâmetros':m27['p'],'gl':m27['gl'],'R²':m27['r2'],'R² ajustado':m27['r2aj'],'RMSE treino':m27['rmse']},{'Plano':'BB (13 pontos)','N':m13['n'],'Parâmetros':m13['p'],'gl':m13['gl'],'R²':m13['r2'],'R² ajustado':m13['r2aj'],'RMSE treino':m13['rmse']}]);tabela('Comparação 27 × 13',resumo)
 pred27=prever(m27,df);predbb=prever(m13,df);v=df[['ID','Y']].copy();v['Predito_3x3x3']=pred27;v['Predito_BB']=predbb;v['Erro_BB']=v.Y-v.Predito_BB;v['Abs_erro_BB']=v.Erro_BB.abs();v['Delta_pred']=v.Predito_BB-v.Predito_3x3x3
 tabs=st.tabs(['Coeficientes','Validação 27 condições','Superfícies e ΔY','Ótimos e decisão'])
 with tabs[0]:
  co=m27['coef'][['Termo','Coeficiente']].merge(m13['coef'][['Termo','Coeficiente']],on='Termo',suffixes=('_27','_BB'));co['Diferença']=co.Coeficiente_BB-co.Coeficiente_27;tabela('Comparação de parâmetros',co)
  st.plotly_chart(px.bar(co,x='Termo',y=['Coeficiente_27','Coeficiente_BB'],barmode='group',title='Coeficientes: 27 versus 13'),use_container_width=True)
 with tabs[1]:
  tabela('Predição BB nas 27 condições (13 treino + 14 fora do treino)',v)
  a,b=st.columns(2)
  with a:
   st.plotly_chart(px.scatter(v,x='Y',y='Predito_BB',title='BB: observado × predito nos 27'),use_container_width=True)
   st.plotly_chart(px.histogram(v,x='Erro_BB',title='Distribuição dos erros BB'),use_container_width=True)
  with b:
   st.plotly_chart(px.bar(v,x='ID',y='Erro_BB',title='Erro BB por condição'),use_container_width=True)
   st.plotly_chart(px.bar(v,x='ID',y='Abs_erro_BB',title='Erro absoluto BB'),use_container_width=True)
  st.metric('RMSE BB nas 27 condições',f"{np.sqrt(np.mean(v.Erro_BB**2)):.4f}")
  st.caption('Os 13 pontos de treino BB estão incluídos nas 27 condições: análise mista, não um conjunto de validação inteiramente independente. Avalie também os 14 pontos não utilizados no treino.')
  leave=v[~v.ID.isin(BB)];st.metric('RMSE BB nos 14 pontos não utilizados',f"{np.sqrt(np.mean(leave.Erro_BB**2)):.4f}")
 with tabs[2]:
  par=st.selectbox('Par de fatores',[('A','S'),('A','P'),('S','P')],format_func=lambda k:' × '.join(k));f=({'A','S','P'}-set(par)).pop();level=st.select_slider(f'{f} fixo',[-1.,0.,1.],value=0.)
  st.subheader('Modelo 3³ completo');superficie(m27,*par,{f:level})
  st.subheader('Modelo Box–Behnken');superficie(m13,*par,{f:level})
  st.subheader('ΔY = previsão BB − previsão 3³');superficie(m13,*par,{f:level},delta=m27)
 with tabs[3]:
  d=extremos(m27,['A','S','P'],resolucao=19);e=extremos(m13,['A','S','P'],resolucao=19)
  tabela('Máximo e mínimo na grade (não validados)',pd.DataFrame([{'Modelo':'3³','Extremo':v,'A':row.A,'S':row.S,'P':row.P,'Predito':row.Predito} for v,row in [('Máximo',d.loc[d.Predito.idxmax()]),('Mínimo',d.loc[d.Predito.idxmin()])]]+[{'Modelo':'BB','Extremo':v,'A':row.A,'S':row.S,'P':row.P,'Predito':row.Predito} for v,row in [('Máximo',e.loc[e.Predito.idxmax()]),('Mínimo',e.loc[e.Predito.idxmin()])]]))
  d['Divergencia']=np.abs(d.Predito-prever(m13,d));cand=d.loc[d.Divergencia.idxmax()];st.write(f'Candidato exploratório a novo ensaio por maior divergência: A={cand.A:.2f}, S={cand.S:.2f}, P={cand.P:.2f}. Não equivale necessariamente ao desenho ótimo de informação.')
 st.download_button('⬇ Baixar comparação completa',v.to_csv(index=False).encode('utf-8-sig'),file_name='CNT735_assunto15.csv')

def cubo(df):
 vertices=df.drop_duplicates(['A','B','C']).sort_values('Condicao')
 fig=go.Figure()
 for _,v in vertices.iterrows():
  for _,w in vertices.iterrows():
   if sum(v[k]!=w[k] for k in ['A','B','C'])==1 and v.Condicao<w.Condicao:
    fig.add_trace(go.Scatter3d(x=[v.A,w.A],y=[v.B,w.B],z=[v.C,w.C],mode='lines',line=dict(color='gray',width=3),showlegend=False,hoverinfo='skip'))
 for lote,g in vertices.groupby('Lote'):
  fig.add_trace(go.Scatter3d(x=g.A,y=g.B,z=g.C,mode='markers+text',text=g.Condicao.astype(str),textposition='top center',name=lote,marker=dict(size=8),customdata=g[['Bloco']],hovertemplate='A=%{x}, B=%{y}, C=%{z}<br>ABC=%{customdata[0]}'))
 fig.update_layout(title='Cubo 2³: vértices classificados por lote (Bloco=ABC)',scene=dict(xaxis_title='A (tempo)',yaxis_title='B (temperatura)',zaxis_title='C (pressão)'))
 st.plotly_chart(fig,use_container_width=True)

def do16():
 st.warning('Força magnética inicialmente **SIMULADA**, não evidência sobre fabricantes. Lote=ABC: o efeito de lote e a interação de terceira ordem são inseparáveis. Um único lote por fabricante tampouco permite inferência geral sobre fabricantes.')
 df=editar('16');df['Bloco']=df.A*df.B*df.C;df['Lote']=np.where(df.Bloco==-1,'Fabricante A','Fabricante B')
 m=ajustar(df,'16');metricas(m)
 tabs=st.tabs(['Contrastes e ANOVA','Gráficos e diagnóstico','Cubo 2³','Interpretação e redesenho'])
 with tabs[0]:passos(m);tabela('Contrastes, efeitos e ANOVA (supõe réplicas independentes)',anova_ortogonal(df,m));grafico_coef(m)
 with tabs[1]:
  st.plotly_chart(px.box(df,x='Lote',y='Y',points='all',title='Força magnética por lote (diferença confundida com ABC)'),use_container_width=True)
  fatores(df,['A','B','C']);interacoes(df,[('A','B'),('A','C'),('B','C')]);
  effects=anova_ortogonal(df,m).iloc[:7];st.plotly_chart(px.bar(effects,x='Termo',y='Efeito',title='Efeitos fatoriais; bloco/ABC confundidos'),use_container_width=True);diagnosticos(m)
 with tabs[2]:cubo(df);tabela('Pertencimento aos blocos',df.drop_duplicates('Condicao')[['Condicao','A','B','C','Bloco','Lote']])
 with tabs[3]:
  st.markdown('**A origem da ferrita produziu alteração detectável?** O contraste lote/ABC pode apresentar significância condicional sob pressupostos de réplicas independentes; não se pode atribuí-lo exclusivamente ao fabricante. ABC foi sacrificado para distribuir as condições entre blocos. Se ABC for fisicamente relevante, a atribuição ao fabricante é indecidível neste desenho. Para separar lote e ABC, aplicar as oito condições em ambos os lotes (16 corridas mínimas, 48 com triplicatas). Para generalizar a fabricantes, amostrar mais de um lote independente de cada um.')
  tabela('Evidência de colinearidade',pd.DataFrame({'Bloco':df.Bloco,'ABC':df.A*df.B*df.C}).head(8));st.caption('Não colocar ABC e Bloco simultaneamente em X: posto deficiente.')
 baixas(df,m,'assunto16')

st.title('CNT735 · Laboratório de Planejamento Experimental')
st.caption('PENt/COPPE/UFRJ · Assuntos 11–16 · matrizes, modelos, inferência crítica e gráficos interativos')
st.sidebar.header('Navegação')
sel=st.sidebar.radio('Escolha o assunto',list(TITULOS),format_func=lambda k:TITULOS[k])
st.sidebar.markdown('**Fluxo**: editar dados → inspecionar modelo → avaliar diagnóstico → interpretar → exportar.')
st.sidebar.caption('Planilhas de aula preservadas como fontes. Nenhuma resposta experimental foi inventada pela aplicação.')
st.subheader(TITULOS[sel])
{'11':do11,'12':do12,'13':do13,'14':do14,'15':do15,'16':do16}[sel]()
