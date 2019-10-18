set.seed(123)
tau_1_tru <- 0.25

x <- y <- rep(0,1000)

for ( i in 1:1000){
	if( runif(1) < tau_1_tru ) {
		x[i] <- rnorm(1, mean=1)
		y[i] <- "heads"
	}else {
		x[i] <- rnorm(1, mean=7)
		y[i] <- "tails"
	}
}

densityPlot(~x, col=as.factor(y))

print( x[1] )

dnorm( x[1], mean=0)

mu_1 <- 0
mu_2 <- 1

tau_1 <- 0.5
tau_2 <- 0.5

for ( i in 1:10) {
	T_1 <- tau_1 * dnorm( x, mu_1)
	T_2 <- tau_2 * dnorm( x, mu_2)


	P_1 <- T_1 / (T_1 + T_2)
	P_2 <- T_2 / (T_1 + T_2)

	tau_1 <- mean(P_1)
	tau_2 <- mean(P_2)

	mu_1 <- sum( P_1 * x) / sum(P_1)
	mu_2 <- sum( P_2 * x) / sum(P_2)


	print( c(mu_1, mu_2, mean(P_1)) )
}