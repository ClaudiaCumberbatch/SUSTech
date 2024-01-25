package exer;

public class InternalCalculate extends Calculate {
    public int calculateParkingPrice(double hours){
        int price = 0;
        price += (int) hours / 24 * 60;
        hours = hours % 24;
        if (hours >= 2 && hours <= 24) {
            price += Math.min(5 * (int) (hours - 2), 60);
        }
        return price;
    }

    @Override
    public int calculatePoints(double hours) {
        return (int) hours / 24 * 2 + 1;
    }
}
