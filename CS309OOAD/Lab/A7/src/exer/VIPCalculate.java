package exer;

public class VIPCalculate extends Calculate {
    public int calculateParkingPrice(double hours){
        int price = 0;
        price += (int) hours / 24 * 100;
        hours = hours % 24;
        if (hours >= 0.5 && hours <= 24) {
            price += Math.min(15 * (int) hours, 100);
        }
        return price;
    }

    @Override
    public int calculatePoints(double hours) {
        return ((int) hours / 24 * 2 + 1) * 2;
    }
}
