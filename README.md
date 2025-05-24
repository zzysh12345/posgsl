This is the code of paper "Prototypical Subgraph Structure Learning for Graph Classification" submitted to CIKM-25.



To run this code, you need to install OpenGSL, a recently proposed library for GSL. Please see their paper and repository for more details.



Example to run experiment:

```python
python main.py --data PROTEINS --method posgsl --config config/posgsl_PROTEINS.yaml
```