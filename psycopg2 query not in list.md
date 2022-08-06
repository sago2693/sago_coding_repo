### Given a list of values
lista_ids = ["1SuM4pgjiynihBsma-boMoL2K-6PKYZbZoh5jjjc",
"1IyzPTwissZaBq4jA0y-IJuqOZMVc0oYUPvrsXQ"]

### The statement where (not) with any allows for the use of the list to filter
db_connection.query("""select * from conteo_siembra_tabla_control
 where not id = ANY (%s)""",(lista_ids,))
